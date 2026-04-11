#!/usr/bin/env python3
"""
WavetableLayer Phase 5 benchmark script.

Drives a live Vivid instance via the control server HTTP API to measure
audio performance of WavetableLayer vs legacy WavetableOsc + VoiceMixer.

Prerequisites:
  - Vivid running with vivid-wavetable linked
  - Control server on port 9876 (default)

Usage:
  python3 benchmarks/wavetable_layer_benchmark.py [--vivid-build-dir PATH] [--port 9876]

Performance gates:
  - Single-instance: mean audio_load <= 0.30
  - Four-instance:   xruns == 0 over measurement window
  - WavetableLayer 4x load < legacy 4x load
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError


def post(port: int, method: str, body: dict | None = None) -> dict:
    url = f"http://127.0.0.1:{port}/{method}"
    data = json.dumps(body or {}).encode()
    req = Request(url, data=data, method="POST",
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def load_graph(port: int, graph_path: str) -> dict:
    return post(port, "load_graph", {"path": graph_path})


def run_diagnostics(port: int) -> dict:
    return post(port, "run_diagnostics")


def detect_backend(vivid_build_dir: Path, package_build_dir: Path) -> str:
    """Infer available renderer backend from build artifacts.

    This is heuristic — the control server does not currently expose
    per-operator backend telemetry, so this reports the strongest backend
    compiled into the package/core build, not the dispatch taken by every
    node on every block.
    """
    cache = package_build_dir / "CMakeCache.txt"
    if platform.system() == "Darwin" and cache.exists():
        text = cache.read_text(errors="ignore")
        if "VIVID_WAVETABLE_PREFER_ACCELERATE:BOOL=ON" in text:
            return "Accelerate preferred (macOS package build; runtime dispatch not exposed)"
        if "VIVID_WAVETABLE_ENABLE_ACCELERATE:BOOL=ON" in text:
            return "Accelerate built but benchmark-gated off (Highway preferred)"

    hwy_lib = vivid_build_dir / "_deps" / "highway-build" / "libhwy.a"
    if hwy_lib.exists():
        return "SIMD (Highway) [build-tree inference]"
    hwy_dylib = vivid_build_dir / "_deps" / "highway-build" / "libhwy.dylib"
    if hwy_dylib.exists():
        return "SIMD (Highway) [build-tree inference]"
    return "scalar [build-tree inference]"


def measure_fixture(port: int, graph_path: str, label: str,
                    settle_seconds: float = 3.0,
                    sample_count: int = 10,
                    sample_interval: float = 0.5) -> dict:
    """Load a graph, wait for steady-state, sample diagnostics."""
    abs_path = str(Path(graph_path).resolve())
    print(f"\n--- {label} ---")
    print(f"Loading: {graph_path}")

    result = load_graph(port, abs_path)
    if not result.get("ok", False):
        print(f"  ERROR: load_graph failed: {result.get('error', 'unknown')}")
        return {"error": result.get("error", "load failed")}

    print(f"  Settling for {settle_seconds}s...")
    time.sleep(settle_seconds)

    loads = []
    xruns_start = None
    top_nodes = []
    buffer_size = 0
    sample_rate = 0

    for i in range(sample_count):
        diag = run_diagnostics(port)
        if not diag.get("ok", False):
            err = diag.get("error", "unknown")
            print(f"  ERROR: run_diagnostics returned ok=false on sample {i}: {err}")
            return {"error": f"diagnostics failed: {err}"}
        health = diag.get("health", {})
        audio = health.get("audio", {})
        if not audio.get("running", False):
            print(f"  ERROR: audio engine not running on sample {i}")
            return {"error": "audio engine not running"}

        load = audio.get("load", 0.0)
        xruns = audio.get("xruns", 0)
        loads.append(load)

        if xruns_start is None:
            xruns_start = xruns
        if i == sample_count - 1:
            xruns_end = xruns
            buffer_size = audio.get("buffer_size", 0)
            sample_rate = audio.get("sample_rate", 0)
            top_nodes = audio.get("top_nodes", [])

        time.sleep(sample_interval)

    xruns_during = (xruns_end or 0) - (xruns_start or 0)
    mean_load = statistics.mean(loads) if loads else 0.0
    max_load = max(loads) if loads else 0.0

    print(f"  audio_load: mean={mean_load:.4f} max={max_load:.4f}")
    print(f"  xruns during measurement: {xruns_during}")
    if top_nodes:
        for node in top_nodes[:5]:
            nid = node.get("node_id", "?")
            ntype = node.get("type", "?")
            ema = node.get("ema_block_us", 0)
            pct = node.get("last_block_budget_pct", 0.0)
            print(f"  top_node: {nid} ({ntype}) ema={ema}us budget_pct={pct:.1f}%")

    return {
        "label": label,
        "mean_load": mean_load,
        "max_load": max_load,
        "xruns": xruns_during,
        "buffer_size": buffer_size,
        "sample_rate": sample_rate,
        "top_nodes": top_nodes,
        "samples": loads,
    }


def main():
    parser = argparse.ArgumentParser(description="WavetableLayer Phase 5 benchmark")
    parser.add_argument("--vivid-build-dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent.parent / "vivid" / "build"),
                        help="Path to vivid core build directory")
    parser.add_argument("--package-build-dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "build"),
                        help="Path to vivid-wavetable package build directory")
    parser.add_argument("--port", type=int, default=9876,
                        help="Vivid control server port")
    parser.add_argument("--settle", type=float, default=3.0,
                        help="Seconds to wait after loading each graph")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of diagnostics samples per fixture")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Number of full benchmark-suite repetitions")
    args = parser.parse_args()

    vivid_build = Path(args.vivid_build_dir)
    package_build = Path(args.package_build_dir)
    pkg_root = Path(__file__).resolve().parent.parent

    # Verify connection
    try:
        diag = run_diagnostics(args.port)
    except (URLError, ConnectionRefusedError, OSError) as e:
        print(f"ERROR: Cannot connect to Vivid on port {args.port}: {e}")
        print("Make sure Vivid is running with vivid-wavetable linked.")
        sys.exit(1)

    backend = detect_backend(vivid_build, package_build)
    machine = f"{platform.machine()} / {platform.system()} {platform.release()}"

    print("=" * 60)
    print("WavetableLayer Benchmark Report")
    print("=" * 60)
    print(f"Machine: {machine}")
    print(f"Backend: {backend}")

    # Fixture paths
    pad_demo = pkg_root / "graphs" / "core" / "wavetable_layer_pad_demo.json"
    layer_stress = pkg_root / "graphs" / "core" / "wavetable_layer_stress.json"
    legacy_stress = pkg_root / "graphs" / "core" / "wavetable_osc_stress.json"

    for path in [pad_demo, layer_stress, legacy_stress]:
        if not path.exists():
            print(f"ERROR: fixture not found: {path}")
            sys.exit(1)

    repeat_count = max(1, args.repeat)
    all_passed = True
    improvements = []
    single_loads = []
    layer_loads = []
    legacy_loads = []

    for run_idx in range(repeat_count):
        if repeat_count > 1:
            print("\n" + "#" * 60)
            print(f"Benchmark repetition {run_idx + 1}/{repeat_count}")
            print("#" * 60)

        single = measure_fixture(args.port, str(pad_demo),
                                 "Single Instance (wavetable_layer_pad_demo)",
                                 settle_seconds=args.settle, sample_count=args.samples)
        layer_4x = measure_fixture(args.port, str(layer_stress),
                                   "Four Instance WavetableLayer (wavetable_layer_stress)",
                                   settle_seconds=args.settle, sample_count=args.samples)
        legacy_4x = measure_fixture(args.port, str(legacy_stress),
                                    "Four Instance Legacy (wavetable_osc_stress)",
                                    settle_seconds=args.settle, sample_count=args.samples)

        print("\n" + "=" * 60)
        print(f"Summary {run_idx + 1}/{repeat_count}")
        print("=" * 60)

        if single.get("buffer_size"):
            print(f"Buffer size: {single['buffer_size']}")
            print(f"Sample rate: {single['sample_rate']}")

        gates_passed = 0
        gates_total = 3

        if "error" not in single:
            single_loads.append(single["mean_load"])
            passed = single["mean_load"] <= 0.30
            status = "PASS" if passed else "FAIL"
            print(f"\nGate 1 — Single instance mean load <= 0.30: {status} ({single['mean_load']:.4f})")
            if passed:
                gates_passed += 1
        else:
            print(f"\nGate 1 — Single instance: SKIP (load failed)")

        if "error" not in layer_4x:
            layer_loads.append(layer_4x["mean_load"])
            passed = layer_4x["xruns"] == 0
            status = "PASS" if passed else "FAIL"
            print(f"Gate 2 — Four instance xruns == 0: {status} ({layer_4x['xruns']} xruns)")
            if passed:
                gates_passed += 1
        else:
            print(f"Gate 2 — Four instance: SKIP (load failed)")

        if "error" not in legacy_4x:
            legacy_loads.append(legacy_4x["mean_load"])

        if "error" not in layer_4x and "error" not in legacy_4x:
            passed = layer_4x["mean_load"] < legacy_4x["mean_load"]
            improvement = ((legacy_4x["mean_load"] - layer_4x["mean_load"])
                           / max(legacy_4x["mean_load"], 0.001)) * 100
            improvements.append(improvement)
            status = "PASS" if passed else "FAIL"
            print(f"Gate 3 — WavetableLayer < Legacy: {status}")
            print(f"  WavetableLayer 4x: {layer_4x['mean_load']:.4f}")
            print(f"  Legacy 4x:         {legacy_4x['mean_load']:.4f}")
            print(f"  Improvement:       {improvement:.1f}%")
            if passed:
                gates_passed += 1
        else:
            print(f"Gate 3 — Comparison: SKIP (load failed)")

        print(f"\nGates: {gates_passed}/{gates_total} passed")
        all_passed = all_passed and (gates_passed == gates_total)

    if repeat_count > 1 and improvements:
        print("\n" + "=" * 60)
        print("Repeated Run Summary")
        print("=" * 60)

        def mean_std(values):
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            return mean, std

        single_mean, single_std = mean_std(single_loads)
        layer_mean, layer_std = mean_std(layer_loads)
        legacy_mean, legacy_std = mean_std(legacy_loads)
        improvement_mean, improvement_std = mean_std(improvements)
        print(f"Single WavetableLayer mean load: {single_mean:.4f} ± {single_std:.4f}")
        print(f"4x WavetableLayer mean load:     {layer_mean:.4f} ± {layer_std:.4f}")
        print(f"4x Legacy mean load:             {legacy_mean:.4f} ± {legacy_std:.4f}")
        print(f"Improvement over legacy:         {improvement_mean:.1f}% ± {improvement_std:.1f}%")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
