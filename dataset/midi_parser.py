# -*- coding: utf-8 -*-
# TODO(kaparoo): Need own parser

import madmom.io.midi as mm_midi
import numpy as np


class MIDIParser(object):
    def __init__(self, fps=20, offset_scale=1.0):
        self.fps = fps
        self.offset_scale = offset_scale

    def process(self, path) -> np.ndarray:
        m = mm_midi.MIDIFile(path)
        notes = np.asarray(sorted(m.notes, key=lambda n: (n[0], n[1] * -1)))
        midi_matrix = notes_to_matrix(
            notes, dt=1.0 / self.fps, offset_scale=self.offset_scale
        )
        return midi_matrix


def notes_to_matrix(notes, dt, offset_scale=1.0):
    """Convert sequence of keys to midi matrix"""

    n_frames = int(np.ceil((notes[-1, 0] + notes[-1, 2]) / dt))
    midi_matrix = np.zeros((128, n_frames), dtype=np.uint8)
    for n in notes:
        onset = int(np.ceil(n[0] / dt))
        offset = int(offset_scale * np.ceil((n[0] + n[2]) / dt))
        midi_pitch = int(n[1])
        midi_matrix[midi_pitch, onset:offset] += 1

    return midi_matrix
