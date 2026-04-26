#!/usr/bin/env python3
"""Phase 3 PR2 — VoiceAllocator graph migrator.

For every graph that wires `VoiceAllocator` to its synths/envelopes/filters,
rewrite it to the new pattern:

  note_source/notes_out → synth/notes_in            (synths run their own
                                                     internal allocator now)
  note_source/notes_out → NoteBreakout/notes_in     (only when shared per-voice
                                                     control state is needed)
  NoteBreakout/voice_gates       → EnvelopeAu/gate
  NoteBreakout/voice_ids         → EnvelopeAu/lane_ids
  NoteBreakout/voice_freqs       → Filter/frequencies, DualFilter/frequencies
  NoteBreakout/voice_velocities  → VoiceMixer/velocities, VoiceDrive/velocities

The VoiceAllocator node is deleted along with all wires into and out of it.

Inventory (run this in vivid-wavetable repo):
    python3 scripts/migrate_voice_allocator.py --check
Apply migration:
    python3 scripts/migrate_voice_allocator.py --apply
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Operators that consume notes_in directly. These get a single
# `note_source/notes_out → synth/notes_in` wire and drop their lane inputs.
SYNTH_TYPES = {
    # vivid-wavetable
    "WavetableOsc", "AnalogOsc", "WavetableLayer", "SubOsc", "NoiseLayer",
    # vivid-core (in case shared graphs reach into them)
    "FmSynth", "Sampler", "SP404", "Slicer",
}

# Per-voice consumers that should be rewired from NoteBreakout/voice_* lanes.
# Mapping: target_input_port_name → NoteBreakout output port.
NOTEBREAKOUT_TARGETS = {
    # EnvelopeAu / Envelope
    "gate":       "voice_gates",
    "lane_ids":   "voice_ids",
    # Filter / DualFilter
    "frequencies": "voice_freqs",
    # VoiceMixer / VoiceDrive
    "velocities": "voice_velocities",
}

# Lane-input ports on synths that are being deleted in PR3. Wires going to
# these on a synth are dropped (the synth picks up the same data from notes_in).
SYNTH_LANE_INPUTS = {
    "frequencies", "gates", "velocities", "lane_ids",
    "pitch_mod", "position_mod", "warp_mod",
}

# Aliases: VoiceAllocator may appear under PolyVoiceAllocator (legacy alias
# already resolved by operator_aliases.cpp, but graphs may still use it).
ALLOCATOR_TYPES = {"VoiceAllocator", "PolyVoiceAllocator"}


def find_allocator(nodes: dict) -> str | None:
    """Return the node id of the first VoiceAllocator-type node, or None."""
    for nid, ndef in nodes.items():
        if ndef.get("type") in ALLOCATOR_TYPES:
            return nid
    return None


def find_note_source(graph: dict, allocator_id: str) -> str | None:
    """Find the upstream node that feeds the allocator's note inputs.

    Looks for connections into `allocator/notes_in` (the lane input). If
    multiple sources feed the allocator (rare), returns the first.
    """
    for c in graph.get("connections", []):
        to = c.get("to", "")
        if to.startswith(allocator_id + "/"):
            port = to.split("/", 1)[1]
            if port in {"notes_in", "notes"}:
                return c["from"].split("/", 1)[0]
    # Fallback: any upstream node feeding the allocator.
    for c in graph.get("connections", []):
        to = c.get("to", "")
        if to.startswith(allocator_id + "/"):
            return c["from"].split("/", 1)[0]
    return None


def already_has_notebreakout(nodes: dict) -> str | None:
    for nid, ndef in nodes.items():
        if ndef.get("type") == "NoteBreakout":
            return nid
    return None


def fresh_node_id(nodes: dict, base: str) -> str:
    if base not in nodes:
        return base
    i = 2
    while f"{base}{i}" in nodes:
        i += 1
    return f"{base}{i}"


def reshape_module_ports(module_def: dict, nb_id: str, log: list[str]) -> None:
    """For module JSON: replace lane-based note inputs with a single `notes_in`
    custom-ref input bound to NoteBreakout/notes_in. The internal graph then
    fans NoteBreakout/notes_out and voice_* to consumers."""
    ports = module_def.setdefault("ports", [])
    new_ports = []
    dropped = []
    for p in ports:
        name = p.get("name", "")
        direction = p.get("direction", "")
        if direction == "input" and name in {"notes", "velocities", "gates"}:
            dropped.append(name)
            continue
        new_ports.append(p)
    # Insert notes_in at the top of inputs (place before the first non-input).
    notes_in_port = {
        "name": "notes_in",
        "type": "custom_ref",
        "data_type": "VividNoteBuffer",
        "direction": "input",
        "bind": f"{nb_id}/notes_in",
    }
    insert_at = 0
    for i, p in enumerate(new_ports):
        if p.get("direction") != "input":
            insert_at = i
            break
        insert_at = i + 1
    new_ports.insert(insert_at, notes_in_port)
    module_def["ports"] = new_ports
    if dropped:
        log.append(f"  ~ module ports: dropped {dropped}, added notes_in (bind {nb_id}/notes_in)")


def migrate_graph(graph: dict) -> tuple[bool, list[str]]:
    """Rewrite the graph in place. Returns (changed, change_log)."""
    nodes = graph.setdefault("nodes", {})
    conns = graph.setdefault("connections", [])
    log: list[str] = []

    allocator_id = find_allocator(nodes)
    if allocator_id is None:
        return False, log

    is_module = isinstance(graph.get("module"), dict)

    if is_module:
        # Modules carry their own external port surface. The "note source" is
        # external — we treat the inserted NoteBreakout as the in-graph proxy
        # and fan from its notes_out to all synths. Modules have one allocator
        # so reusing an existing NoteBreakout (if present) is fine.
        nb_id = already_has_notebreakout(nodes) or fresh_node_id(nodes, "voice_breakout")
        if nb_id not in nodes:
            nodes[nb_id] = {"type": "NoteBreakout"}
            log.append(f"  + insert NoteBreakout '{nb_id}' (module proxy for external notes_in)")
        src = nb_id
        src_port = "notes_out"
        src_type = "NoteBreakout (module passthrough)"
    else:
        s = find_note_source(graph, allocator_id)
        if s is None:
            log.append(f"WARN: no upstream note source for allocator {allocator_id}")
            return False, log
        src = s
        src_port = "notes_out"
        src_type = nodes.get(src, {}).get("type", "?")

    log.append(f"allocator={allocator_id} note_source={src} ({src_type})")

    # Inventory consumers of the allocator's outputs.
    consumers: dict[str, list[tuple[str, str]]] = {}  # consumer_node → [(va_port, dst_port)]
    for c in conns:
        if c.get("from", "").startswith(allocator_id + "/"):
            dst_node, dst_port = c["to"].split("/", 1)
            va_port = c["from"].split("/", 1)[1]
            consumers.setdefault(dst_node, []).append((va_port, dst_port))

    synth_consumers = [n for n in consumers if nodes.get(n, {}).get("type") in SYNTH_TYPES]
    other_consumers = [n for n in consumers if nodes.get(n, {}).get("type") not in SYNTH_TYPES]

    # Modules already have a NoteBreakout (it's the module-internal proxy for
    # the external notes_in port), so they always "need" one. Otherwise only
    # graphs with non-synth consumers need the breakout.
    needs_breakout = is_module or bool(other_consumers)
    if is_module:
        nb_id = src  # already set above
    else:
        nb_id = None
        if needs_breakout:
            # Always create a FRESH NoteBreakout per allocator. Graphs with
            # multiple parallel voice chains (e.g., wavetable_layer_stress
            # has voices_1/2/3/4 with separate ChordProgression sources) need
            # one NoteBreakout per chain — they consume different note streams.
            base = "voice_breakout"
            # If this allocator's id has a numeric/textual suffix, mirror it.
            for sep in ("_", "-"):
                if sep in allocator_id:
                    suffix = allocator_id.split(sep, 1)[1]
                    base = f"voice_breakout_{suffix}"
                    break
            nb_id = fresh_node_id(nodes, base)
            nodes[nb_id] = {"type": "NoteBreakout"}
            log.append(f"  + insert NoteBreakout '{nb_id}'")

    # New connection list, built incrementally.
    new_conns: list[dict] = []

    # 1. Drop ALL wires into and out of the allocator.
    for c in conns:
        f = c.get("from", "")
        t = c.get("to", "")
        if f.startswith(allocator_id + "/") or t.startswith(allocator_id + "/"):
            continue
        new_conns.append(c)
    log.append(f"  - drop {len(conns) - len(new_conns)} wire(s) touching {allocator_id}")

    # 2. Drop wires going into synth lane-input ports (those synths get notes_in).
    pre_synth_drop = len(new_conns)
    new_conns_after_synth_drop = []
    for c in new_conns:
        t = c.get("to", "")
        if "/" in t:
            tn, tp = t.split("/", 1)
            if tn in synth_consumers and tp in SYNTH_LANE_INPUTS:
                continue  # drop
        new_conns_after_synth_drop.append(c)
    new_conns = new_conns_after_synth_drop
    if pre_synth_drop != len(new_conns):
        log.append(f"  - drop {pre_synth_drop - len(new_conns)} legacy lane-input wire(s) on synths")

    # 3. Rewire connections from upstream source's legacy lane outputs (notes/
    #    velocities/gates) into the now-defunct allocator-lane-input pattern.
    #    Some graphs have e.g. `chords/velocities → osc/velocities` directly
    #    (bypassing the allocator). The synths we own no longer accept those —
    #    they get velocity from notes_in. Drop those if they target a synth.
    pre_src_drop = len(new_conns)
    new_conns2 = []
    for c in new_conns:
        f = c.get("from", "")
        t = c.get("to", "")
        if "/" in f and "/" in t:
            sn, sp = f.split("/", 1)
            tn, tp = t.split("/", 1)
            if (not is_module) and sn == src and tn in synth_consumers and tp in SYNTH_LANE_INPUTS:
                continue  # drop legacy direct-lane wire to a synth
        new_conns2.append(c)
    new_conns = new_conns2
    if pre_src_drop != len(new_conns):
        log.append(f"  - drop {pre_src_drop - len(new_conns)} legacy direct-lane wire(s) from {src}")

    # 4. Wire src/<src_port> → synth/notes_in for every synth consumer.
    for sc in sorted(synth_consumers):
        wire = {"from": f"{src}/{src_port}", "to": f"{sc}/notes_in"}
        if wire not in new_conns:
            new_conns.append(wire)
            log.append(f"  + {src}/{src_port} → {sc}/notes_in")

    # 5. For graph-mode (not module), wire the upstream source into the
    #    inserted NoteBreakout. In module-mode src IS the NoteBreakout, so
    #    there's no internal wire — the external notes_in port is bound to
    #    NoteBreakout/notes_in via reshape_module_ports below.
    if needs_breakout and not is_module:
        wire = {"from": f"{src}/{src_port}", "to": f"{nb_id}/notes_in"}
        if wire not in new_conns:
            new_conns.append(wire)
            log.append(f"  + {src}/{src_port} → {nb_id}/notes_in")

    # 6. If module: reshape its external port surface to a single notes_in
    #    custom_ref input bound to NoteBreakout/notes_in.
    if is_module:
        reshape_module_ports(graph["module"], nb_id, log)

    if needs_breakout:
        # 7. For each non-synth consumer that pulled a va_port → dst_port wire,
        for cnode in sorted(other_consumers):
            ctype = nodes.get(cnode, {}).get("type", "?")
            for (va_port, dst_port) in consumers[cnode]:
                # Map by destination port name first (filter/frequencies,
                # envelope/gate, envelope/lane_ids, voice_mixer/velocities, ...).
                nb_port = NOTEBREAKOUT_TARGETS.get(dst_port)
                if nb_port is None:
                    # Fallback: map by source allocator port name.
                    fallback = {
                        "frequencies": "voice_freqs",
                        "gates":       "voice_gates",
                        "velocities":  "voice_velocities",
                        "lane_ids":    "voice_ids",
                    }.get(va_port)
                    nb_port = fallback
                if nb_port is None:
                    log.append(f"  WARN: no rewrite for {allocator_id}/{va_port} → {cnode}/{dst_port}")
                    continue
                wire = {"from": f"{nb_id}/{nb_port}", "to": f"{cnode}/{dst_port}"}
                if wire not in new_conns:
                    new_conns.append(wire)
                    log.append(f"  + {nb_id}/{nb_port} → {cnode}/{dst_port}")

    # 7. Delete the allocator node.
    nodes.pop(allocator_id, None)
    log.append(f"  - delete node '{allocator_id}'")

    graph["connections"] = new_conns

    # 8. PR1 cleanup: SubOsc/NoiseLayer's per-voice multichannel `output` is
    #    now `voices_out` (stereo `output` is the new summed bus). Rewrite any
    #    wires that fed VoiceMixer/VoiceDrive's per-voice inputs.
    PER_VOICE_AUDIO_INPUTS = {("VoiceMixer", "input"), ("VoiceDrive", "input")}
    SOURCE_TYPES_NEEDING_VOICES_OUT = {"SubOsc", "NoiseLayer"}
    rewires = 0
    for c in graph["connections"]:
        f = c.get("from", "")
        t = c.get("to", "")
        if "/" not in f or "/" not in t:
            continue
        sn, sp = f.split("/", 1)
        tn, tp = t.split("/", 1)
        s_type = nodes.get(sn, {}).get("type")
        t_type = nodes.get(tn, {}).get("type")
        if (sp == "output" and s_type in SOURCE_TYPES_NEEDING_VOICES_OUT
                and (t_type, tp) in PER_VOICE_AUDIO_INPUTS):
            c["from"] = f"{sn}/voices_out"
            rewires += 1
    if rewires:
        log.append(f"  ~ rewire {rewires} SubOsc/NoiseLayer per-voice output connection(s) to voices_out")

    return True, log


# Module types whose external port surface was reshaped from
# notes/velocities/gates (lane) to a single notes_in (custom_ref).
MIGRATED_MODULE_TYPES = {
    "HybridKeys", "LayerPad", "DualWavetablePad", "SubAirPad", "GlassInteractionKeys",
}

# Per-note expression external ports on GlassInteractionKeys (pressures,
# slides). These bind to non-VoiceAllocator internal nodes (expr_cv) and
# stay alive on the module surface.
EXPRESSION_PORTS = {"pressures", "slides"}


def rewire_module_consumers(graph: dict) -> tuple[bool, list[str]]:
    """For graphs (not modules themselves) that wire `something/notes`,
    `/velocities`, `/gates` into a migrated-module instance, drop those wires
    and add a single `something/notes_out → instance/notes_in` connection.
    Other inputs (like pressures/slides) stay intact.
    """
    if isinstance(graph.get("module"), dict):
        return False, []

    nodes = graph.get("nodes", {})
    conns = graph.get("connections", [])
    log: list[str] = []

    # Group note-ish connections by (target_node, source_node).
    pairs: dict[tuple[str, str], list[dict]] = {}
    other_conns: list[dict] = []
    LANE_PORTS = {"notes", "velocities", "gates"}
    for c in conns:
        f = c.get("from", "")
        t = c.get("to", "")
        if "/" in f and "/" in t:
            sn, _ = f.split("/", 1)
            tn, tp = t.split("/", 1)
            if (nodes.get(tn, {}).get("type") in MIGRATED_MODULE_TYPES
                    and tp in LANE_PORTS):
                pairs.setdefault((tn, sn), []).append(c)
                continue
        other_conns.append(c)

    if not pairs:
        return False, log

    new_conns = list(other_conns)
    for (tn, sn), bundle in pairs.items():
        # Drop bundle, add single notes_out → notes_in wire.
        new_wire = {"from": f"{sn}/notes_out", "to": f"{tn}/notes_in"}
        if new_wire not in new_conns:
            new_conns.append(new_wire)
        log.append(f"  ~ rewire {sn}/(notes,velocities,gates) → {tn}/* "
                   f"to {sn}/notes_out → {tn}/notes_in")

    graph["connections"] = new_conns
    return True, log


def find_graphs() -> list[Path]:
    out = []
    for sub in ("graphs", "modules", "archive/graphs"):
        d = REPO / sub
        if d.is_dir():
            out.extend(sorted(d.rglob("*.json")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Write changes back to disk (default is dry-run).")
    ap.add_argument("--check", action="store_true",
                    help="List graphs that still reference VoiceAllocator (CI mode).")
    args = ap.parse_args()

    files = find_graphs()
    changed: list[tuple[Path, list[str]]] = []
    skipped: list[Path] = []

    for path in files:
        try:
            with path.open() as f:
                graph = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            skipped.append(path)
            print(f"SKIP {path.relative_to(REPO)}: {e}", file=sys.stderr)
            continue

        # Some graphs have multiple VoiceAllocator nodes (e.g., voices_1,
        # voices_2, voices_3 driving parallel voice chains). Loop until none
        # remain. Each iteration migrates one allocator + its consumers.
        all_log: list[str] = []
        any_ok = False
        for _ in range(16):  # bounded — no graph has >16 allocators
            ok, log = migrate_graph(graph)
            if not ok:
                break
            any_ok = True
            all_log.extend(log)

        # Module-consumer rewires: graphs that instantiate a migrated module
        # and feed its now-defunct lane inputs need their wires updated to
        # the new notes_in custom_ref port.
        consumer_ok, consumer_log = rewire_module_consumers(graph)
        if consumer_ok:
            any_ok = True
            all_log.extend(consumer_log)

        if any_ok:
            changed.append((path, all_log))
            if args.apply:
                with path.open("w") as f:
                    json.dump(graph, f, indent=2)
                    f.write("\n")

    if args.check:
        # Re-scan for any remaining VoiceAllocator references in JSON text.
        offenders = []
        for path in files:
            try:
                with path.open() as f:
                    if any(t in f.read() for t in ALLOCATOR_TYPES):
                        offenders.append(path)
            except OSError:
                pass
        if offenders:
            print("FAIL: graphs still reference VoiceAllocator:")
            for p in offenders:
                print(f"  {p.relative_to(REPO)}")
            return 1
        print("OK: no graph references VoiceAllocator")
        return 0

    print(f"Migrated {len(changed)} graph(s)" + ("" if args.apply else " (dry run)"))
    for path, log in changed:
        print(f"\n{path.relative_to(REPO)}")
        for line in log:
            print(f"  {line}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
