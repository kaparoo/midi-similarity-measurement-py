# -*- coding: utf-8 -*-
# TODO(kaparoo): Rewrite error messages more properly

import enum
import numbers
import numpy as np
from typing import Callable, Tuple, Union

from .constant import MAX_MIDI_KEY, MIN_MIDI_KEY, NUM_MIDI_KEYS, NUM_PITCH_CLASSES

__all__ = ["MIDIUnit"]


class _Type(enum.Enum):
    Note = (0,)
    Rest = (1,)


class MIDIUnit(object):

    __slots__ = ["_type", "_midi_key", "_velocity"]

    def __init__(self, midi_key: int, velocity: float) -> None:
        if not isinstance(midi_key, numbers.Integral):
            raise TypeError(type(midi_key))
        if not isinstance(velocity, numbers.Real):
            raise TypeError(type(velocity))

        if MAX_MIDI_KEY >= midi_key >= MIN_MIDI_KEY and velocity >= 0.0:
            self._type = _Type.Note
            self._midi_key = int(midi_key)
            self._velocity = float(velocity)
        else:
            self._type = _Type.Rest
            self._midi_key = MAX_MIDI_KEY + 1
            self._velocity = 0.0

    def is_note(self) -> bool:
        return self._type == _Type.Note

    def is_rest(self) -> bool:
        return self._type == _Type.Rest

    def get_midi_key(self) -> Union[int, None]:
        if self.is_note():
            return self._midi_key

    def get_velocity(self) -> Union[float, None]:
        if self.is_note():
            return self._velocity

    def get_values(self) -> Union[Tuple[int, float], None]:
        if self.is_note():
            return self._midi_key, self._velocity

    def __str__(self) -> str:
        if self.is_note():
            return "Note(%d, %.4f)" % (self._midi_key, self._velocity)
        else:
            return "Rest"

    @classmethod
    def new_rest(cls):
        cls.__init__(MAX_MIDI_KEY + 1, 0)


class MIDIUnitSequence(list):
    def __init__(self, *args) -> None:
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


class MIDIUnitSequenceList(list):
    def __init__(self, *args) -> None:
        super(MIDIUnitSequenceList, self).__init__()

    @staticmethod
    def from_midi_matrix(
        midi_matrix: np.ndarray,
        onset_weight: float = 10.0,
        decay_fn: Callable[[float], float] = lambda x: 1,
    ):
        # assert isinstance(midi_matrix, np.ndarray)
        # assert len(midi_matrix) == NUM_MIDI_KEYS
        # assert len(midi_matrix.shape) <= 2
        # assert isinstance(onset_weight, (int, float)) and onset_weight > 0

        midi_unit_sequence_list = MIDIUnitSequenceList()
        midi_matrix = np.reshape(midi_matrix, [NUM_MIDI_KEYS, -1]).T
        for frame_idx in range(len(midi_matrix)):
            midi_unit_sequence = MIDIUnitSequence()
            for midi_key in range(NUM_MIDI_KEYS):
                if midi_matrix[frame_idx, midi_key] <= 0:
                    continue
                elif frame_idx > 0:
                    prev_velocity = midi_matrix[frame_idx - 1, midi_key]
                    if prev_velocity > 0:
                        velocity = decay_fn(prev_velocity)
                    else:
                        velocity = onset_weight
                else:
                    velocity = onset_weight
                midi_matrix[frame_idx, midi_key] = velocity
                midi_unit_sequence.append(MIDIUnit(midi_key, velocity))
            if len(midi_unit_sequence) == 0:
                midi_unit_sequence.append(MIDIUnit(MAX_MIDI_KEY + 1, 0))
            midi_unit_sequence_list.append(midi_unit_sequence)

        return midi_unit_sequence_list

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

    def to_pitch_histogram(self, normalize: bool = True) -> np.ndarray:
        histogram = np.zeros([NUM_PITCH_CLASSES], dtype=np.float32)
        for midi_unit_sequence in self:
            for midi_unit in midi_unit_sequence:
                if midi_unit.is_note():
                    midi_key, velocity = midi_unit.get_values()
                    histogram[midi_key % NUM_PITCH_CLASSES] += velocity
        if normalize:
            histogram /= np.sum(histogram + 1e-7)
        return histogram

    def to_significant_unit_sequence(
        self, unit_metric: Callable[[MIDIUnitSequence], MIDIUnit] = lambda x: x[-1]
    ) -> MIDIUnitSequence:
        sequence = MIDIUnitSequence()
        for midi_unit_sequence in self:
            significant_unit = unit_metric(midi_unit_sequence)
            sequence.append(significant_unit)
        if not sequence:
            sequence.append(MIDIUnit.new_rest())
        return sequence
