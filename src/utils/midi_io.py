"""MIDI I/O utilities (stubs).

Implementations use `pretty_midi` for full functionality.
"""
from typing import List
import pretty_midi
import os


def write_midi_from_events(events: List[int], out_path: str):
    # placeholder: create an empty MIDI file
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    pm.instruments.append(piano)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pm.write(out_path)


def read_midi(path: str):
    return pretty_midi.PrettyMIDI(path)
