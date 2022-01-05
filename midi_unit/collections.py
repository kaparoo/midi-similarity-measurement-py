# -*- coding: utf-8 -*-

from __future__ import annotations

import abc
import numbers
import numpy as np
from typing import SupportsIndex, Tuple, Union

try:
    from constants import MAX_MIDI_KEY, MIN_MIDI_KEY, NUM_MIDI_KEYS, NUM_PITCHES
    from functions import DecayFn
except ImportError:
    from .constants import MAX_MIDI_KEY, MIN_MIDI_KEY, NUM_MIDI_KEYS, NUM_PITCHES
    from .functions import DecayFn


__all__ = ["MIDIUnit", "MIDINote", "MIDIRest", "MIDIUnitSequence", "MIDIUnitSeqList"]


class MIDIUnit(abc.ABC):

    __slots__ = ["_midi_key", "_velocity"]

    def __init__(self) -> None:
        raise NotImplementedError()

    @property
    def midi_key(self) -> int:
        return self._midi_key

    @property
    def velocity(self) -> float:
        return self._velocity

    @property
    def values(self) -> Tuple[int, float]:
        return (self._midi_key, self._velocity)

    def __repr__(self) -> str:
        return self.__str__()

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def is_note(self) -> bool:
        raise NotImplementedError()


class MIDINote(MIDIUnit):
    def __init__(self, midi_key: int, velocity: float) -> None:
        if not isinstance(midi_key, numbers.Integral):
            raise TypeError()
        elif not MIN_MIDI_KEY <= midi_key <= MAX_MIDI_KEY:
            raise ValueError()
        if not isinstance(velocity, numbers.Real):
            raise TypeError()
        elif velocity < 0.0:
            raise ValueError()
        self._midi_key = int(midi_key)
        self._velocity = float(velocity)

    def __str__(self) -> str:
        return f"Note({self.midi_key}, {self.velocity})"

    def is_note(self) -> bool:
        return True


class MIDIRest(MIDIUnit):
    def __new__(cls) -> MIDIRest:
        if not hasattr(cls, "_instance"):
            cls._instance = super(MIDIRest, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self._midi_key = MAX_MIDI_KEY + 1
        self._velocity = 0.0

    def __str__(self) -> str:
        return "Rest"

    def is_note(self) -> bool:
        return False


class MIDIUnitSequence(list):
    def __init__(self) -> None:
        super(MIDIUnitSequence, self).__init__()

    def append(self, unit: Union[MIDINote, MIDIRest]) -> None:
        if not isinstance(unit, (MIDINote, MIDIRest)):
            raise TypeError()
        super(MIDIUnitSequence, self).append(unit)

    def extend(self, sequence: MIDIUnitSequence) -> None:
        if not isinstance(sequence, MIDIUnitSequence):
            raise TypeError()
        super(MIDIUnitSequence, self).extend(sequence)

    def to_midi_matrix(self, set_velocity: bool = False) -> np.ndarray:
        matrix = np.zeros((NUM_MIDI_KEYS, len(self)), dtype=np.float32)
        for frame_idx, unit in enumerate(self):
            if unit.is_note():
                midi_key, velocity = unit.values
                matrix[midi_key, frame_idx] = velocity if set_velocity else 1
        return matrix

    def __str__(self) -> str:
        units = [unit.__str__() for unit in self]
        return "[" + ", ".join(units) + "]"

    def __repr__(self) -> str:
        return self.__str__()

    def __getitem__(
        self, val: Union[slice, SupportsIndex]
    ) -> Union[Union[MIDINote, MIDIRest], MIDIUnitSequence]:
        if isinstance(val, SupportsIndex):
            return super(MIDIUnitSequence, self).__getitem__(val)
        elif isinstance(val, slice):
            sequence = MIDIUnitSequence()
            for unit in super(MIDIUnitSequence, self).__getitem__(val):
                sequence.append(unit)
            return sequence
        raise TypeError()


class MIDIUnitSeqList(list):

    __slots__ = ["extraction_policy"]

    def __init__(self) -> None:
        super(MIDIUnitSeqList, self).__init__()

    def append(self, sequence: MIDIUnitSequence) -> None:
        if not isinstance(sequence, MIDIUnitSequence):
            raise TypeError()
        super(MIDIUnitSeqList, self).append(sequence)

    def extend(self, seqlist: MIDIUnitSeqList) -> None:
        if not isinstance(seqlist, MIDIUnitSeqList):
            raise TypeError()
        super(MIDIUnitSeqList, self).extend(seqlist)

    def __str__(self) -> str:
        seqlist = [sequence.__str__() for sequence in self]
        return "[" + ", ".join(seqlist) + "]"

    def __repr__(self) -> str:
        return self.__str__()

    def __getitem__(
        self, val: Union[slice, SupportsIndex]
    ) -> Union[MIDIUnitSequence, MIDIUnitSeqList]:
        if isinstance(val, SupportsIndex):
            return super(MIDIUnitSeqList, self).__getitem__(val)
        elif isinstance(val, slice):
            seqlist = MIDIUnitSeqList()
            for sequence in super(MIDIUnitSeqList, self).__getitem__(val):
                seqlist.append(sequence)
            return seqlist
        raise TypeError()

    @staticmethod
    def from_midi_matrix(
        matrix: np.ndarray, decay_fn: DecayFn = lambda x: x
    ) -> MIDIUnitSeqList:
        seqlist = MIDIUnitSeqList()
        decayed_matrix = np.zeros_like(matrix)
        for midi_key in range(NUM_MIDI_KEYS):
            decayed_matrix[midi_key, :] = decay_fn(matrix[midi_key, :])
        for frame_idx in range(matrix.shape[-1]):
            sequence = MIDIUnitSequence()
            for midi_key in range(NUM_MIDI_KEYS):
                if (velocity := decayed_matrix[midi_key, frame_idx]) > 0:
                    sequence.append(MIDINote(midi_key, velocity))
            if not sequence:
                sequence.append(MIDIRest())
            seqlist.append(sequence)
        return seqlist

    def to_midi_matrix(self, set_velocity: bool = False) -> np.ndarray:
        matrix = np.zeros((NUM_MIDI_KEYS, len(self)), dtype=np.float32)
        for frame_idx, frame in enumerate(self):
            for unit in frame:
                if unit.is_note():
                    midi_key, velocity = unit.values
                    matrix[midi_key, frame_idx] = velocity if set_velocity else 1
        return matrix

    @property
    def pitch_histogram(self) -> np.ndarray:
        histogram = np.zeros(NUM_PITCHES, dtype=np.float32)
        for frame in self:
            for unit in frame:
                if unit.is_note():
                    midi_key, velocity = unit.values
                    histogram[midi_key % NUM_PITCHES] += velocity
        if (total := np.sum(histogram)) > 0:
            histogram /= total
        return histogram

    @property
    def repr_sequence(self) -> MIDIUnitSequence:
        sequence = MIDIUnitSequence()
        for frame in self:
            repr_unit = frame[0]
            for curr_unit in frame[1:]:
                if repr_unit.velocity <= curr_unit.velocity:
                    repr_unit = curr_unit
            sequence.append(repr_unit)
        if not sequence:
            sequence.append(MIDIRest())
        return sequence

    @property
    def highest_sequence(self) -> MIDIUnitSequence:
        sequence = MIDIUnitSequence()
        for frame in self:
            repr_unit = frame[0]
            for curr_unit in frame[1:]:
                if repr_unit.velocity <= curr_unit.velocity:
                    repr_unit = curr_unit
            sequence.append(repr_unit)
        if not sequence:
            sequence.append(MIDIRest())
        return sequence

    @property
    def lowest_sequence(self) -> MIDIUnitSequence:
        sequence = MIDIUnitSequence()
        for frame in self:
            repr_unit = frame[0]
            for curr_unit in frame[1:]:
                if repr_unit.velocity >= curr_unit.velocity:
                    repr_unit = curr_unit
            sequence.append(repr_unit)
        if not sequence:
            sequence.append(MIDIRest())
        return sequence
