# -*- coding: utf-8 -*-

try:
    from .constant import MAX_MIDI_KEY, MIN_MIDI_KEY, NUM_MIDI_KEYS, NUM_PITCH_CLASSES
    from .decay_fn import DecayFn
    from .unit import MIDIRest, MIDIUnit, MIDIUnitSequence, MIDIUnitSequenceList
except ImportError:
    from constant import MAX_MIDI_KEY, MIN_MIDI_KEY, NUM_MIDI_KEYS, NUM_PITCH_CLASSES
    from decay_fn import DecayFn
    from unit import MIDIRest, MIDIUnit, MIDIUnitSequence, MIDIUnitSequenceList

__all__ = [
    "MAX_MIDI_KEY",
    "MIN_MIDI_KEY",
    "NUM_MIDI_KEYS",
    "NUM_PITCH_CLASSES",
    "DecayFn",
    "MIDIRest",
    "MIDIUnit",
    "MIDIUnitSequence",
    "MIDIUnitSequenceList",
]
