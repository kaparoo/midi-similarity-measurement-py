# -*- coding: utf-8 -*-

import midi
import numpy as np
from typing import Tuple

try:
    from algorithm import CostFn, compare_midi_key, dtw, euclidean
except ImportError:
    from .algorithm import CostFn, compare_midi_key, dtw, euclidean

__all__ = ["measure"]


def measure(
    source_matrix: np.ndarray,
    target_matrix: np.ndarray,
    compensation_frame: int = 0,
    cost_fn: CostFn = compare_midi_key,
    decay_fn: midi.DecayFn = lambda x: x,
    subsequence: bool = False,
) -> Tuple[float, float]:
    source = midi.MIDIUnitSequenceList.from_midi_matrix(source_matrix, decay_fn)
    target = midi.MIDIUnitSequenceList.from_midi_matrix(target_matrix, decay_fn)
    source.compensation_frame = compensation_frame
    target.compensation_frame = compensation_frame

    source_sequence = source.repr_unit_sequence
    target_sequence = target.repr_unit_sequence
    timewarping_distance, (head, tail), _, _ = dtw(
        source_sequence, target_sequence, cost_fn, subsequence
    )

    source_histogram = source.pitch_histogram
    target_histogram = target[head : tail + 1].pitch_histogram
    histogram_distance = euclidean(source_histogram, target_histogram)

    return histogram_distance, timewarping_distance
