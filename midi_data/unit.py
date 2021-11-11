# -*- coding: utf-8 -*-

from enum import Enum
from numbers import Integral, Real
from typing import Tuple, Union

from .constant import MAX_MIDI_KEY, MIN_MIDI_KEY

__all__ = ["MIDIUnit"]


class _Type(Enum):
    Note = (0,)
    Rest = (1,)


class MIDIUnit(object):

    __slots__ = ["_type", "_midi_key", "_velocity"]

    def __init__(self, midi_key: int, velocity: float) -> None:
        # TODO(kaparoo): Rewrite error messages more properly
        if not isinstance(midi_key, Integral):
            raise TypeError(type(midi_key))
        if not isinstance(velocity, Real):
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
