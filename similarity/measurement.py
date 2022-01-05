# -*- coding: utf-8 -*-

import midi_unit
import numpy as np
import time
from typing import Dict, Tuple, Union

try:
    from algorithm import CostFn, compare_midi_key, dtw, euclidean
except ImportError:
    from .algorithm import CostFn, compare_midi_key, dtw, euclidean

__all__ = ["measure", "Similarity"]

Similarity = Tuple[float, float, float]


def measure(
    source_matrix: np.ndarray,
    target_matrix: np.ndarray,
    cost_fn: CostFn = compare_midi_key,
    decay_fn: midi_unit.DecayFn = lambda x: x,
    subsequence: bool = False,
    measure_time: bool = False,
) -> Union[Similarity, Tuple[Similarity, Dict[str, float]]]:
    if measure_time:
        timestamp1 = time.time()

    source = midi_unit.MIDIUnitSeqList.from_midi_matrix(source_matrix, decay_fn)
    target = midi_unit.MIDIUnitSeqList.from_midi_matrix(target_matrix, decay_fn)

    if measure_time:
        timestamp2 = time.time()

    source_sequence = source.repr_sequence
    target_sequence = target.repr_sequence
    timewarping_distance, (head, tail), _, _ = dtw(
        source_sequence, target_sequence, cost_fn, subsequence
    )

    if measure_time:
        timestamp3 = time.time()

    source_histogram = source.pitch_histogram
    target_histogram = target[head : tail + 1].pitch_histogram
    histogram_distance = euclidean(source_histogram, target_histogram)

    length_ratio = len(target) / len(source)

    similarity = (histogram_distance, timewarping_distance, length_ratio)

    if measure_time:
        timestamp4 = time.time()
        execution_times = {
            "midi_matrix": timestamp2 - timestamp1,
            "timewarping": timestamp3 - timestamp2,
            "others": timestamp4 - timestamp3,
            "total": timestamp4 - timestamp1,
        }
        return similarity, execution_times
    else:
        return similarity
