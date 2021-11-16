# -*- coding: utf-* -*-

import function
import midi
import numpy as np
from typing import Tuple

__all__ = ["euclidean", "levenshtein", "dtw"]


def euclidean(source_histogram: np.ndarray, target_histogram: np.ndarray) -> float:
    # TODO(kaparoo): Rewrite assertions shortly
    assert isinstance(source_histogram, np.ndarray)
    assert len(source_histogram) == midi.NUM_PITCH_CLASSES
    assert len(source_histogram.shape) == 1
    assert isinstance(target_histogram, np.ndarray)
    assert len(target_histogram) == midi.NUM_PITCH_CLASSES
    assert len(target_histogram.shape) == 1
    return np.sum((source_histogram - target_histogram) ** 2) ** 0.5


def levenshtein(
    source_sequence: midi.MIDIUnitSequence,
    target_sequence: midi.MIDIUnitSequence,
    cost_metric: function.CostMetric = function.compare_cost_fn,
) -> float:
    assert isinstance(source_sequence, midi.MIDIUnitSequence)
    assert isinstance(target_sequence, midi.MIDIUnitSequence)
    source_len, target_len = len(source_sequence), len(target_sequence)
    accumulated_cost_matrix = np.zeros([source_len, target_len], dtype=np.float32)

    for i, s in enumerate(source_sequence):
        for j, t in enumerate(target_sequence):
            if i == 0 and j == 0:
                pass
            elif i == 0:
                accumulated_cost_matrix[0, j] = j
            elif j == 0:
                accumulated_cost_matrix[i, 0] = i
            else:
                accumulated_cost_matrix[i, j] = min(
                    [
                        accumulated_cost_matrix[i - 1, j] + 1,
                        accumulated_cost_matrix[i, j - 1] + 1,
                        accumulated_cost_matrix[i - 1, j - 1] + cost_metric(s, t),
                    ]
                )

    return (
        accumulated_cost_matrix[source_len - 1, target_len - 1]
        / (source_len * target_len) ** 0.5
    )


def dtw(
    source_sequence: midi.MIDIUnitSequence,
    target_sequence: midi.MIDIUnitSequence,
    cost_metric: function.CostMetric = function.compare_cost_fn,
) -> float:
    assert isinstance(source_sequence, midi.MIDIUnitSequence)
    assert isinstance(target_sequence, midi.MIDIUnitSequence)
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

    return (
        accumulated_cost_matrix[source_len - 1, target_len - 1]
        / (source_len * target_len) ** 0.5
    )


def subsequence_matching(
    source_sequence: midi.MIDIUnitSequence,
    target_sequence: midi.MIDIUnitSequence,
    cost_metric: function.CostMetric = function.compare_cost_fn,
) -> Tuple[float, Tuple[int, int]]:
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

    deltas = accumulated_cost_matrix[source_len - 1, :]
    # tail = target_len - 1
    # cost = deltas[tail]
    # # idx = tail
    # while idx >= 0:
    #     if deltas[idx] < cost:
    #         tail = idx
    #         cost = deltas[idx]
    tail = np.argmin(deltas)
    cost = deltas[tail]
    head = 0
    i, j = source_len - 1, tail
    while i >= 0 and j >= 0:
        if i == 0:
            head = j
            break
        elif j == 0:
            i -= 1
        else:
            horizontal_cost = accumulated_cost_matrix[i][j - 1]
            diagonal_cost = accumulated_cost_matrix[i - 1][j - 1]
            vertical_cost = accumulated_cost_matrix[i - 1][j]
            if horizontal_cost < diagonal_cost:
                if horizontal_cost < vertical_cost:
                    j -= 1
                else:
                    i -= 1
            else:
                i -= 1
                if diagonal_cost <= vertical_cost:
                    j -= 1

    return cost, (head, tail), accumulated_cost_matrix, deltas
