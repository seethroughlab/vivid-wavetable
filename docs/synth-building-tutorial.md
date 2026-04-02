# Building a Synth with vivid-wavetable

Step-by-step tutorial. Test audio at each stage — if a step produces no sound or wrong sound, that's where the bug is.

## Stage 1: Minimal sound — one oscillator to output

```
new_graph
add_node type=WavetableOsc pkg=vivid-wavetable id=osc
add_node type=audio_out id=out
connect from=osc/output to=out/input
```

Set the osc to play a fixed tone:
```
set_param node=osc param=amplitude value=0.8
set_param node=osc param=wavetable value=0
set_param node=osc param=position value=0.5
```

The osc needs frequencies and gates via lane ports. Without a PolyVoiceAllocator, you can test with a Keyboard node or by checking if the osc produces silence (expected — no frequency input yet).

**Expected**: Silence (no frequency/gate input). If you hear noise or a crash, something is wrong with the osc output buffer.

## Stage 2: Add voice allocation

```
add_node type=ChordProgression id=chords
add_node type=Clock id=clock
add_node type=PolyVoiceAllocator pkg=vivid-wavetable id=voices

connect from=clock/beat_phase to=chords/beat_phase
connect from=chords/notes to=voices/notes_in
connect from=chords/velocities to=voices/velocities_in
connect from=chords/gates to=voices/gates_in
connect from=voices/frequencies to=osc/frequencies
connect from=voices/gates to=osc/gates
connect from=voices/velocities to=osc/velocities
```

**Expected**: You should hear a 3-note chord changing every few beats. The osc outputs raw N-channel audio (one channel per voice) directly to audio_out. It will be loud and unfiltered. If you only hear 1 note, the multi-voice output isn't working.

## Stage 3: Add the VoiceMixer

The VoiceMixer takes N-channel per-voice audio and mixes it to stereo with panning and envelope.

```
disconnect from=osc/output to=out/input

add_node type=VoiceMixer pkg=vivid-wavetable id=mixer
add_node type=Envelope id=amp_env

connect from=osc/output to=mixer/input
connect from=voices/gates to=amp_env/gate
connect from=amp_env/value to=mixer/amp_env
connect from=voices/velocities to=mixer/velocities
connect from=mixer/output to=out/input

set_param node=amp_env param=attack value=0.01
set_param node=amp_env param=sustain value=0.8
```

**Expected**: Same chord but now with stereo spread, velocity sensitivity, and a fast attack envelope. Volume will be lower than Stage 2 because the mixer normalizes. If silence, the VoiceMixer lane indices or buffer_has_signal check may be wrong.

## Stage 4: Add a filter

The seed Filter auto-dups to process each voice channel independently.

```
disconnect from=osc/output to=mixer/input

add_node type=Filter id=filter
add_node type=Envelope id=filt_env

connect from=osc/output to=filter/input
connect from=voices/frequencies to=filter/frequencies
connect from=filter/output to=mixer/input

connect from=voices/gates to=filt_env/gate
connect from=filt_env/value to=filter/cutoff_mod

set_param node=filter param=cutoff value=2000
set_param node=filter param=resonance value=0.3
set_param node=filt_env param=attack value=0.5
set_param node=filt_env param=decay value=0.8
set_param node=filt_env param=sustain value=0.3
```

**Expected**: Filtered chord with a sweep on each note attack. If volume drops to near-zero, the filter isn't passing the N-channel audio through correctly. Compare with Stage 3 to isolate.

## Stage 5: Add reverb

```
disconnect from=mixer/output to=out/input

add_node type=Reverb id=reverb

connect from=mixer/output to=reverb/input
connect from=reverb/output to=out/input

set_param node=reverb param=room_size value=0.5
set_param node=reverb param=mix value=0.25
```

**Expected**: Same as Stage 4 but with reverb tail. If the reverb makes it silent, check the mixer stereo output format.

## Stage 6: Add modulation (lane-rate)

```
add_node type=LFO id=pos_lfo
set_param node=pos_lfo param=frequency value=0.15
set_param node=pos_lfo param=amplitude value=0.3

connect from=pos_lfo/value to=osc/position_mod
```

**Expected**: The wavetable position slowly sweeps, changing the timbre. This uses the lane modulation port (control_float → lane), not the audio-rate port.

## Diagnosis checklist

At each stage, run `analyze_output mode=audio window_seconds=3` and check:

| Stage | Expected rms | If too quiet | If silent |
|-------|-------------|--------------|-----------|
| 1 | 0 (no input) | N/A | Correct |
| 2 | > 0.05 | Check osc amplitude | Check frequency/gate connections |
| 3 | > 0.02 | Check mixer normalization | Check mixer lane indices [3,4,5] or buffer_has_signal |
| 4 | > 0.01 | Check filter cutoff | Filter may not pass N-channel audio |
| 5 | > 0.01 | Check reverb mix | Check mixer stereo output |
| 6 | Same as 5 | N/A | Check LFO→lane connection |
