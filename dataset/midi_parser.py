# -*- coding: utf-8 -*-

import madmom.io.midi as mm_midi
import numpy as np
from os import PathLike


__all__ = ["MIDIParser"]


class MIDIParser(object):

    __slot__ = ["_fps", "_scale"]

    def __init__(self, fps: int = 20, note_scale: float = 1.0):
        self._fps = fps
        self._scale = note_scale

    @property
    def fps(self) -> int:
        return self._fps

    @fps.setter
    def fps(self, val: int) -> None:
        self._fps = val

    @property
    def note_scale(self) -> float:
        return self._scale

    @note_scale.setter
    def note_scale(self, val: float) -> None:
        self._scale = val

    def process(self, path: PathLike) -> np.ndarray:
        midi = mm_midi.MIDIFile(path)
        notes = np.asarray(sorted(midi.notes, key=lambda n: (n[0], n[1] * -1)))
        dt = 1.0 / self._fps
        num_frames = int(np.ceil((notes[-1, 0] + notes[-1, 2]) / dt))
        midi_matrix = np.zeros((128, num_frames), dtype=np.uint8)
        for note in notes:
            onset = int(np.ceil(note[0] / dt))
            offset = int(np.ceil((note[0] + self._scale * note[2]) / dt))
            midi_pitch = int(note[1])
            midi_matrix[midi_pitch, onset:offset] += 1
        return midi_matrix
