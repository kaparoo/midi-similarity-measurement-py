# -*- coding: utf-8 -*-

try:
    from constant import MAX_MIDI_KEY, MIN_MIDI_KEY, NUM_MIDI_KEYS, NUM_PITCH_CLASSES
    from decay import DecayFn, get_decay_fn
    from unit import MIDIRest, MIDIUnit, MIDIUnitSequence, MIDIUnitSequenceList
except ImportError:
    from .constant import MAX_MIDI_KEY, MIN_MIDI_KEY, NUM_MIDI_KEYS, NUM_PITCH_CLASSES
    from .decay import DecayFn, get_decay_fn
    from .unit import MIDIRest, MIDIUnit, MIDIUnitSequence, MIDIUnitSequenceList

__all__ = [
    "MAX_MIDI_KEY",
    "MIN_MIDI_KEY",
    "NUM_MIDI_KEYS",
    "NUM_PITCH_CLASSES",
    "DecayFn",
    "get_decay_fn",
    "MIDIRest",
    "MIDIUnit",
    "MIDIUnitSequence",
    "MIDIUnitSequenceList",
]
