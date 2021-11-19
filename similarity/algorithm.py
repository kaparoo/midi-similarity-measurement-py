# -*- coding: utf-* -*-

import midi
import numpy as np
import scipy.signal as signal
from typing import Callable, List, Tuple

__all__ = ["euclidean", "levenshtein", "dtw"]


def euclidean(source_histogram: np.ndarray, target_histogram: np.ndarray) -> float:
    return np.sqrt(np.sum((source_histogram - target_histogram) ** 2))


CostMetric = Callable[[midi.MIDIUnit, midi.MIDIUnit], float]


def compare_cost_fn(s: midi.MIDIUnit, t: midi.MIDIUnit) -> float:
    return float(s.get_midi_key() != t.get_midi_key())


def dtw(
    source_sequence: midi.MIDIUnitSequence,
    target_sequence: midi.MIDIUnitSequence,
    cost_metric: CostMetric = compare_cost_fn,
    stabilize: bool = True,
) -> float:
    if not isinstance(source_sequence, midi.MIDIUnitSequence):
        raise TypeError(type(source_sequence))
    if not isinstance(target_sequence, midi.MIDIUnitSequence):
        raise TypeError(type(target_sequence))

    source_len, target_len = len(source_sequence), len(target_sequence)
    accumulated_cost_matrix = np.zeros([source_len, target_len], dtype=np.float32)

    for i, s in enumerate(source_sequence):
        for j, t in enumerate(target_sequence):
            if i == 0 and j == 0:
                cost = 0
            elif i == 0:
                cost = accumulated_cost_matrix[0, j - 1]
            elif j == 0:
                cost = accumulated_cost_matrix[i - 1, 0]
            else:
                cost = min(
                    [
                        accumulated_cost_matrix[i - 1, j],
                        accumulated_cost_matrix[i, j - 1],
                        accumulated_cost_matrix[i - 1, j - 1],
                    ]
                )
            accumulated_cost_matrix[i, j] = cost_metric(s, t) + cost

    cost = accumulated_cost_matrix[source_len - 1, target_len - 1]
    if stabilize:
        cost = cost / (source_len * target_len) ** 0.5

    return cost


def subsequence_dtw(
    source_sequence: midi.MIDIUnitSequence,
    target_sequence: midi.MIDIUnitSequence,
    cost_metric: CostMetric = compare_cost_fn,
    stabilize: bool = True,
) -> Tuple[
    float, Tuple[int, int], Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]
]:
    if not isinstance(source_sequence, midi.MIDIUnitSequence):
        raise TypeError(type(source_sequence))
    if not isinstance(target_sequence, midi.MIDIUnitSequence):
        raise TypeError(type(target_sequence))

    # Make accumulated cost matrix
    source_len, target_len = len(source_sequence), len(target_sequence)
    accumulated_cost_matrix = np.zeros([source_len, target_len], dtype=np.float32)
    for i, s in enumerate(source_sequence):
        for j, t in enumerate(target_sequence):
            if i == 0:
                accumulated_cost = 0
            elif j == 0:
                accumulated_cost = accumulated_cost_matrix[i - 1, 0]
            else:
                accumulated_cost = min(
                    [
                        accumulated_cost_matrix[i - 1, j],
                        accumulated_cost_matrix[i, j - 1],
                        accumulated_cost_matrix[i - 1, j - 1],
                    ]
                )
            accumulated_cost_matrix[i, j] = cost_metric(s, t) + accumulated_cost

    # Find tail index of objective subsequence from delta functions
    delta_functions = accumulated_cost_matrix[source_len - 1, :]
    reversed_delta = delta_functions[::-1]
    tail = target_len - np.argmin(reversed_delta) - 1
    cost = delta_functions[tail]

    # Find head index of objective subsequence by using back tracking
    head = 0
    i, j = source_len - 1, tail
    optimal_warping_path: List[Tuple[int, int]] = []
    while i >= 0 and j >= 0:
        optimal_warping_path.append((i, j))
        if i == 0:
            head = j
            break
        elif j == 0:
            i -= 1
        else:
            cost_v = accumulated_cost_matrix[i - 1][j]
            cost_h = accumulated_cost_matrix[i][j - 1]
            cost_d = accumulated_cost_matrix[i - 1][j - 1]
            if cost_h < cost_d:
                if cost_h < cost_v:
                    j -= 1
                else:
                    i -= 1
            else:
                i -= 1
                if cost_d <= cost_v:
                    j -= 1
    optimal_warping_path[::-1]  # monotonically increasing

    if stabilize:
        subsequence_len = tail - head + 1
        cost = cost / (source_len * subsequence_len) ** 0.5

    return (
        cost,
        (head, tail),
        (optimal_warping_path, accumulated_cost_matrix, delta_functions),
    )
