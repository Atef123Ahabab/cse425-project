"""MIDI I/O utilities.

Implementations use `pretty_midi` for full functionality.
"""
from typing import List
import pretty_midi
import os
import json


def tokens_to_midi(tokens: List[int], out_path: str, time_bin: float = 0.05, max_shift_bins: int = 100):
    """Convert token sequence back to MIDI file."""
    # Load vocabulary
    vocab_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'vocab.json')
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    # Create inverse vocab
    inv_vocab = {v: k for k, v in vocab.items()}
    
    # Parse tokens
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)  # Piano
    
    current_time = 0.0
    note_on = False
    current_pitch = 0
    note_start_time = 0.0
    
    for token in tokens:
        token_name = inv_vocab.get(token, "")
        
        if token_name.startswith("time_shift_"):
            bins = int(token_name.split("_")[-1])
            current_time += bins * time_bin
            
        elif token_name.startswith("note_on_"):
            # If we were playing a note, finish it
            if note_on:
                duration = current_time - note_start_time
                if duration > 0:
                    piano.notes.append(pretty_midi.Note(
                        velocity=64,  # Default velocity
                        pitch=current_pitch,
                        start=note_start_time,
                        end=current_time
                    ))
            
            # Start new note
            current_pitch = int(token_name.split("_")[-1])
            note_start_time = current_time
            note_on = True
    
    # Finish any remaining note
    if note_on:
        duration = current_time - note_start_time + 0.5  # Add some duration
        piano.notes.append(pretty_midi.Note(
            velocity=64,
            pitch=current_pitch,
            start=note_start_time,
            end=current_time + duration
        ))
    
    pm.instruments.append(piano)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pm.write(out_path)


def write_midi_from_events(events: List[int], out_path: str):
    # placeholder: create an empty MIDI file
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    pm.instruments.append(piano)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pm.write(out_path)


def read_midi(path: str):
    return pretty_midi.PrettyMIDI(path)
