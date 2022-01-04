# -*- coding: utf-8 -*-

"""A parser that parses the midi file to a numpy array."""

import math
import mido
import numpy as np
from os import PathLike
from pathlib import Path
from typing import Tuple


__all__ = ["MIDIParser"]


class MIDIParser(object):
    def __new__(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = super(MIDIParser, cls).__new__(cls)
        return cls._instance

    def __call__(
        self,
        path: PathLike,
        fps: int = 20,
        mark_onset: bool = False,
        use_channel: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not Path(path).is_file():
            raise FileNotFoundError(f"No such midi file: '{path}'")
        if not isinstance(fps, int):
            raise TypeError("`fps` must be a positive integer")
        elif fps <= 0:
            raise ValueError("`fps` must be a positive integer")

        delta_time, cache, notes = 0, {}, []
        for msg in mido.MidiFile(path):
            delta_time += msg.time
            note_on = msg.type == "note_on"
            note_off = msg.type == "note_off"
            if not (note_on or note_off):
                continue
            pitch, velocity = msg.note, msg.velocity
            if use_channel:
                pitch += 128 * (channel := msg.channel)
            if note_on and velocity > 0:
                cache[pitch] = (delta_time, velocity)
            elif note_off or (note_on and velocity == 0):
                if pitch in cache:
                    onset_time, velocity = cache[pitch]
                    duration = delta_time - onset_time
                    if not use_channel:
                        notes.append((pitch, onset_time, duration, velocity))
                    else:
                        pitch_ = pitch % 128
                        notes.append((pitch_, onset_time, duration, velocity, channel))
                    del cache[pitch]
        notes.sort(key=lambda note: (note[1], -note[0]))  # lowest onset, highest pitch

        # TODO(kaparoo): need axis for channel?
        num_frames = math.ceil(fps * (notes[-1][1] + notes[-1][2])) + 1
        midi_matrix = np.zeros((128, num_frames), dtype=np.uint8)
        for pitch, onset_time, duration, *_ in notes:
            onset = math.ceil(fps * onset_time)
            offset = math.ceil(fps * (onset_time + duration))
            midi_matrix[pitch, onset:offset] = 1
            if mark_onset:
                midi_matrix[pitch, onset] = 2

        return np.array(notes), midi_matrix

    @staticmethod
    def parse(
        path: PathLike,
        fps: int = 20,
        mark_onset: bool = False,
        use_channel: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return MIDIParser().__call__(path, fps, mark_onset, use_channel)
