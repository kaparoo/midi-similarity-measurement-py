# -*- coding: utf-8 -*-

import numbers
import numpy as np
from typing import SupportsIndex, Tuple, Union

try:
    from constant import *
    from decay import *
except ImportError:
    from .constant import *
    from .decay import *

__all__ = ["MIDIUnit", "MIDIRest", "MIDIUnitSequence", "MIDIUnitSequenceList"]


class MIDIUnit(object):

    __slots__ = ["_midi_key", "_velocity"]

    def __init__(self, midi_key: int, velocity: float) -> None:
        if not isinstance(midi_key, numbers.Integral):
            raise TypeError(type(midi_key))
        if not isinstance(velocity, numbers.Real):
            raise TypeError(type(velocity))
        self._midi_key = int(midi_key)
        self._velocity = float(velocity)

    @staticmethod
    def new_note(midi_key: int, velocity: float) -> "MIDIUnit":
        if not (MAX_MIDI_KEY >= midi_key >= MIN_MIDI_KEY and velocity > 0):
            raise ValueError(f"({midi_key=:d}, {velocity=:.2f})")
        return MIDIUnit(midi_key, velocity)

    def is_note(self) -> bool:
        midi_key, velocity = self._midi_key, self._velocity
        if MAX_MIDI_KEY >= midi_key >= MIN_MIDI_KEY and velocity > 0:
            return True
        return False

    @property
    def midi_key(self) -> int:
        return self._midi_key

    @property
    def velocity(self) -> float:
        return self._velocity

    @property
    def values(self) -> Tuple[int, float]:
        return (self._midi_key, self._velocity)

    def __str__(self) -> str:
        if self.is_note():
            return f"Note({self._midi_key:d}, {self._velocity:.2f})"
        return "Rest"

    def __repr__(self) -> str:
        return self.__str__()


# Singleton-like constant
MIDIRest = MIDIUnit(midi_key=MAX_MIDI_KEY + 1, velocity=0.0)


class MIDIUnitSequence(list):
    def __init__(self) -> None:
        super(MIDIUnitSequence, self).__init__()

    def append(self, unit: MIDIUnit) -> None:
        if not isinstance(unit, MIDIUnit):
            raise TypeError(type(unit))
        super(MIDIUnitSequence, self).append(unit)

    def extend(self, sequence) -> None:
        if not isinstance(sequence, type(self)):
            raise TypeError(type(sequence))
        super(MIDIUnitSequence, self).extend(sequence)

    def __str__(self) -> str:
        units = [unit.__str__() for unit in self]
        return "[" + ", ".join(units) + "]"

    def __repr__(self) -> str:
        return self.__str__()


class MIDIUnitSequenceList(list):
    def __init__(self, compensation: int = 0) -> None:
        self.compensation = compensation
        super(MIDIUnitSequenceList, self).__init__()

    def append(self, sequence: MIDIUnitSequence) -> None:
        if not isinstance(sequence, MIDIUnitSequence):
            raise TypeError(type(sequence))
        super(MIDIUnitSequenceList, self).append(sequence)

    def extend(self, sequence_list) -> None:
        if not isinstance(sequence_list, type(self)):
            raise TypeError(type(sequence_list))
        super(MIDIUnitSequenceList, self).extend(sequence_list)

    def __str__(self) -> str:
        sequences = [sequence.__str__() for sequence in self]
        return "[" + ", ".join(sequences) + "]"

    def __repr__(self) -> str:
        return self.__str__()

    def __getitem__(self, val: Union[slice, SupportsIndex]):
        if isinstance(val, slice):
            lst = super(MIDIUnitSequenceList, self).__getitem__(val)
            rtn = MIDIUnitSequenceList()
            for e in lst:
                rtn.append(e)
            return rtn
        else:
            return super(MIDIUnitSequenceList, self).__getitem__(val)

    @staticmethod
    def from_midi_matrix(
        matrix: np.ndarray, decay_fn: DecayFn = lambda x: x
    ) -> "MIDIUnitSequenceList":
        sequnece_list = MIDIUnitSequenceList()
        decayed_matrix = np.zeros_like(matrix)
        for midi_key in range(NUM_MIDI_KEYS):
            decayed_matrix[midi_key, :] = decay_fn(matrix[midi_key, :])
        for frame_idx in range(matrix.shape[-1]):
            sequence = MIDIUnitSequence()
            for midi_key in range(NUM_MIDI_KEYS):
                velocity = decayed_matrix[midi_key, frame_idx]
                if velocity > 0:
                    sequence.append(MIDIUnit.new_note(midi_key, velocity))
            if not sequence:
                sequence.append(MIDIRest)
            sequnece_list.append(sequence)
        return sequnece_list

    def to_midi_matrix(self, set_velocity: bool = False) -> np.ndarray:
        num_frames = len(self)
        matrix = np.zeros((NUM_MIDI_KEYS, num_frames), dtype=np.float32)
        for frame_idx, sequence in enumerate(self):
            for unit in sequence:
                if not unit.is_note():
                    break
                midi_key, velocity = unit.values
                if not set_velocity:
                    velocity = 1
                matrix[midi_key, frame_idx] = velocity
        return matrix

    @property
    def compensation_frame(self) -> int:
        return self.compensation

    @compensation_frame.setter
    def compensation_frame(self, val: int) -> None:
        self.compensation = val

    @property
    def pitch_histogram(self) -> np.ndarray:
        histogram = np.zeros([NUM_PITCH_CLASSES], dtype=np.float32)
        for sequence in self:
            for unit in sequence:
                if unit.is_note():
                    midi_key, velocity = unit.values
                    histogram[midi_key % NUM_PITCH_CLASSES] += velocity
        if (total := np.sum(histogram)) > 0:
            histogram /= total
        return histogram

    @property
    def repr_sequence(self) -> MIDIUnitSequence:
        sequence = MIDIUnitSequence()
        for curr_sequence in self:
            repr_unit = curr_sequence[0]
            for curr_unit in curr_sequence[1:]:
                compensated_velocity = curr_unit.velocity + self.compensation
                if repr_unit.velocity <= compensated_velocity:
                    repr_unit = curr_unit
            sequence.append(repr_unit)
        if not sequence:
            sequence.append(MIDIRest)
        return sequence
