# -*- coding: utf-8 -*-

try:
    from annotation import Annotation
    from generator import Dataset, new_generator
    from midi_parser import MIDIParser
except ImportError:
    from .annotation import Annotation
    from .generator import Dataset, new_generator
    from .midi_parser import MIDIParser


__all__ = ["Annotation", "Dataset", "MIDIParser", "new_generator"]
