# -*- coding: utf-* -*-

import midi_unit
import numpy as np
from typing import Callable, Sequence, Tuple


__all__ = [
    "CostFn",
    "compare_midi_key",
    "dtw",
    "euclidean",
    "global_dtw",
    "subsequence_dtw",
]


CostFn = Callable[[midi_unit.MIDIUnit, midi_unit.MIDIUnit], float]


def compare_midi_key(s: midi_unit.MIDIUnit, t: midi_unit.MIDIUnit) -> float:
    return float(s.midi_key != t.midi_key)


def global_dtw(
    source_sequence: midi_unit.MIDIUnitSequence,
    target_sequence: midi_unit.MIDIUnitSequence,
    cost_fn: CostFn = compare_midi_key,
) -> Tuple[float, Tuple[int, int], np.ndarray, Sequence[Tuple[int, int]]]:
    source_len, target_len = len(source_sequence), len(target_sequence)
    cost_matrix = np.zeros([source_len, target_len], dtype=np.float32)

    for i, s in enumerate(source_sequence):
        for j, t in enumerate(target_sequence):
            if i == 0 and j == 0:
                cost = 0
            elif i == 0:
                cost = cost_matrix[0, j - 1]
            elif j == 0:
                cost = cost_matrix[i - 1, 0]
            else:
                cost = min(
                    [
                        cost_matrix[i - 1, j],
                        cost_matrix[i, j - 1],
                        cost_matrix[i - 1, j - 1],
                    ]
                )
            cost_matrix[i, j] = cost_fn(s, t) + cost

    source_tail, target_tail = source_len - 1, target_len - 1
    cost = cost_matrix[source_tail, target_tail]
    cost = cost / np.sqrt(source_len * target_len)

    warping_path = []
    y, x = source_tail, target_tail
    while True:
        warping_path.append((x, y))
        if x == 0 and y == 0:
            break
        elif x == 0:
            y -= 1
        elif y == 0:
            x -= 1
        else:
            costs = [
                cost_matrix[y, x - 1],
                cost_matrix[y - 1, x - 1],
                cost_matrix[y - 1, x],
            ]
            movements = ((-1, 0), (-1, -1), (0, -1))
            dx, dy = movements[np.argmin(costs)]
            x += dx
            y += dy

    return cost, (0, target_tail), cost_matrix, warping_path[::-1]


def subsequence_dtw(
    source_sequence: midi_unit.MIDIUnitSequence,
    target_sequence: midi_unit.MIDIUnitSequence,
    cost_fn: CostFn = compare_midi_key,
) -> Tuple[float, Tuple[int, int], np.ndarray, Sequence[Tuple[int, int]]]:
    source_len, target_len = len(source_sequence), len(target_sequence)
    cost_matrix = np.zeros([source_len, target_len], dtype=np.float32)

    for i, s in enumerate(source_sequence):
        for j, t in enumerate(target_sequence):
            if i == 0:
                cost = 0
            elif j == 0:
                cost = cost_matrix[i - 1, 0]
            else:
                cost = min(
                    [
                        cost_matrix[i - 1, j],
                        cost_matrix[i, j - 1],
                        cost_matrix[i - 1, j - 1],
                    ]
                )
            cost_matrix[i, j] = cost_fn(s, t) + cost

    delta_functions = cost_matrix[source_len - 1, :]
    subsequence_tail = target_len - np.argmin(delta_functions[::-1]) - 1
    cost = delta_functions[subsequence_tail]

    warping_path = []
    subsequence_head = 0
    y, x = source_len - 1, subsequence_tail
    while True:
        warping_path.append((x, y))
        if y == 0:
            subsequence_head = x
            break
        elif x == 0:
            y -= 1
        else:
            costs = [
                cost_matrix[y, x - 1],
                cost_matrix[y - 1, x - 1],
                cost_matrix[y - 1, x],
            ]
            movements = ((-1, 0), (-1, -1), (0, -1))
            dx, dy = movements[np.argmin(costs)]
            x += dx
            y += dy

    subsequence_len = subsequence_tail - subsequence_head + 1
    cost = cost / np.sqrt(source_len * subsequence_len)

    return cost, (subsequence_head, subsequence_tail), cost_matrix, warping_path[::-1]


def dtw(
    source_sequence: midi_unit.MIDIUnitSequence,
    target_sequence: midi_unit.MIDIUnitSequence,
    cost_fn: CostFn = compare_midi_key,
    subsequence: bool = False,
) -> Tuple[float, Tuple[int, int], np.ndarray, Sequence[Tuple[int, int]]]:
    if subsequence:
        return subsequence_dtw(source_sequence, target_sequence, cost_fn)
    else:
        return global_dtw(source_sequence, target_sequence, cost_fn)


def euclidean(source_histogram: np.ndarray, target_histogram: np.ndarray) -> float:
    return np.sqrt(np.sum((source_histogram - target_histogram) ** 2))
