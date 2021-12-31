# -*- coding: utf-8 -*-

from typing import Tuple
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

    def process(
        self, path: PathLike, mark_onset: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        midi = mm_midi.MIDIFile(path)
        notes = np.asarray(sorted(midi.notes, key=lambda n: (n[0], n[1] * -1)))
        onsets = []
        num_frames = int(np.ceil((notes[-1, 0] + notes[-1, 2]) * self._fps))
        midi_matrix = np.zeros((128, num_frames), dtype=np.uint8)
        for note in notes:
            onset = int(np.ceil(note[0] * self._fps))
            offset = int(np.ceil((note[0] + self._scale * note[2]) * self._fps))
            midi_pitch = int(note[1])
            midi_matrix[midi_pitch, onset:offset] = 1
            if mark_onset:
                midi_matrix[midi_pitch, onset] = 2
            onsets.append(onset)
        onsets = np.sort(np.asarray(onsets)).astype(np.float32)
        return midi_matrix, notes, onsets


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    midi_parser = MIDIParser(fps=20, note_scale=1.0)
    midi_path = (
        "../dataset/newbie-dataset/Clementi/sonatina_op36_no3_pt1/0_wcpark_1.mid"
    )

    matrix1, _, _ = midi_parser.process(midi_path, mark_onset=False)
    matrix2, _, _ = midi_parser.process(midi_path, mark_onset=True)

    plt.figure(figsize=(16, 8))

    plt.subplot(2, 1, 1)
    plt.title("Without Marking")
    plt.imshow(
        matrix1[:, 500:700],
        cmap="gray",
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    plt.ylabel("MIDI Key")
    plt.xlabel("Frame")
    plt.ylim([0, 127])
    plt.xlim([0, 199])

    plt.subplot(2, 1, 2)
    plt.title("With Marking")
    plt.imshow(
        matrix2[:, 500:700],
        cmap="gray",
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    plt.ylabel("MIDI Key")
    plt.xlabel("Frame")
    plt.ylim([0, 127])
    plt.xlim([0, 199])

    plt.tight_layout()
    plt.show()
