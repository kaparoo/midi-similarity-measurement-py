# -*- coding: utf-8 -*-

from madmom.io.midi import MIDIFile
import numpy as np
from os import PathLike


__all__ = ["MIDIParser"]


class MIDIParser(object):
    def __init__(self, fps: int = 20, scale: float = 1.0):
        self.fps = fps
        self.scale = scale

    def process(self, path: PathLike) -> np.ndarray:
        midi = MIDIFile(path)
        notes = np.asarray(sorted(midi.notes, key=lambda n: (n[0], n[1] * -1)))
        dt = 1.0 / self.fps
        num_frames = int(np.ceil((notes[-1, 0] + notes[-1, 2]) / dt))
        midi_matrix = np.zeros((128, num_frames), dtype=np.uint8)
        for note in notes:
            onset = int(np.ceil(note[0] / dt))
            offset = int(np.ceil((note[0] + self.scale * note[2]) / dt))
            midi_pitch = int(note[1])
            midi_matrix[midi_pitch, onset:offset] += 1
        return midi_matrix
