# -*- coding: utf-8 -*-

from enum import Enum
from typing import Tuple, Union

if __name__ == "__main__":
    from constant import MAX_MIDI_KEY
else:
    from .constant import MAX_MIDI_KEY

__all__ = ["MIDIUnit"]


class _Type(Enum):
    Note = 0,
    Rest = 1,


class MIDIUnit(object):

    __slots__ = ["_type", "_midi_key", "_velocity"]

    def __init__(self, midi_key: int, velocity: float) -> None:
        if MAX_MIDI_KEY >= midi_key >= 0 and velocity >= 0.0:
            self._type = _Type.Note
            self._midi_key = midi_key
            self._velocity = velocity
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
            return f"Note({self._midi_key}, {self._velocity})"
        else:
            return u"Rest"


if __name__ == "__main__":
    unit1 = MIDIUnit(12, 3.14)
    print(unit1)  # Note(12, 3.14)

    unit2 = MIDIUnit(137, 0.0)
    print(unit2)  # Rest

    unit3 = MIDIUnit(37, -124)
    print(unit3)  # Rest
