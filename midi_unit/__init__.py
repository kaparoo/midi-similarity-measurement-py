# -*- coding: utf-8 -*-

try:
    from collections import (
        MIDIUnit,
        MIDINote,
        MIDIRest,
        MIDIUnitSequence,
        MIDIUnitSeqList,
    )
    from constants import MAX_MIDI_KEY, MIN_MIDI_KEY, NUM_MIDI_KEYS, NUM_PITCHES
    from functions import DecayFn, make_decay_fn
except ImportError:
    from .collections import (
        MIDIUnit,
        MIDINote,
        MIDIRest,
        MIDIUnitSequence,
        MIDIUnitSeqList,
    )
    from .constants import MAX_MIDI_KEY, MIN_MIDI_KEY, NUM_MIDI_KEYS, NUM_PITCHES
    from .functions import DecayFn, make_decay_fn

__all__ = [
    "MAX_MIDI_KEY",
    "MIN_MIDI_KEY",
    "NUM_MIDI_KEYS",
    "NUM_PITCHES",
    "MIDIUnit",
    "MIDINote",
    "MIDIRest",
    "MIDIUnitSequence",
    "MIDIUnitSeqList",
    "DecayFn",
    "make_decay_fn",
]
