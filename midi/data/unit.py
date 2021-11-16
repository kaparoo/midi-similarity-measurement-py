# -*- coding: utf-8 -*-
# TODO(kaparoo): Rewrite error messages more properly

import enum
import numbers
import numpy as np
from typing import List, Tuple, Union

from .constant import MAX_MIDI_KEY, MIN_MIDI_KEY, NUM_MIDI_KEYS, NUM_PITCH_CLASSES

__all__ = ["MIDIUnit", "MIDIUnitSequence", "MIDIUnitSequenceList"]


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

        if MAX_MIDI_KEY >= midi_key >= MIN_MIDI_KEY and velocity > 0:
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

    @staticmethod
    def new_rest():
        return MIDIUnit(MAX_MIDI_KEY + 1, 0.0)


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
        midi_matrix: np.ndarray, settling_frame: int,
    ):
        prev_pressed: List[bool] = [False] * NUM_MIDI_KEYS
        midi_matrix = np.reshape(midi_matrix, [NUM_MIDI_KEYS, -1]).T
        midi_unit_sequence_list = MIDIUnitSequenceList()
        for frame_idx in range(len(midi_matrix)):
            midi_unit_sequence = MIDIUnitSequence()
            for midi_key in range(NUM_MIDI_KEYS):
                velocity = 0
                if midi_matrix[frame_idx, midi_key] <= 0:
                    prev_pressed[midi_key] = False
                elif frame_idx > 0:
                    prev_velocity = midi_matrix[frame_idx - 1, midi_key]
                    if prev_velocity > 0:
                        velocity = prev_velocity - 1
                    elif not prev_pressed[midi_key]:
                        velocity = settling_frame - 1
                    prev_pressed[midi_key] = True
                else:
                    velocity = settling_frame - 1
                    prev_pressed[midi_key] = True
                midi_matrix[frame_idx, midi_key] = velocity
                if velocity > 0:
                    midi_unit_sequence.append(MIDIUnit(midi_key, velocity))
            if len(midi_unit_sequence) == 0:
                midi_unit_sequence.append(MIDIUnit.new_rest())
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

    def __getitem__(self, val):
        if isinstance(val, slice):
            lst = super(MIDIUnitSequenceList, self).__getitem__(val)
            rtn = MIDIUnitSequenceList()
            for e in lst:
                rtn.append(e)
            return rtn
        else:
            return super(MIDIUnitSequenceList, self).__getitem__(val)

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

    def to_representative_unit_sequence(
        self, compensation_frame: int = 0
    ) -> MIDIUnitSequence:
        sequence = MIDIUnitSequence()
        for midi_unit_sequence in self:
            representative_unit = midi_unit_sequence[0]
            for midi_unit in midi_unit_sequence[1:]:
                if representative_unit.get_velocity() <= (
                    compensation_frame + midi_unit.get_velocity()
                ):
                    representative_unit = midi_unit
            sequence.append(representative_unit)
        if not sequence:
            sequence.append(MIDIUnit.new_rest())
        return sequence
