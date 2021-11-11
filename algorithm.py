# -*- coding: utf-* -*-

import function
import midi
import numpy as np

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
    cost_metric: function.DistanceMetric = function.get_distance_metric(),
) -> float:
    assert isinstance(source_sequence, midi.MIDIUnitSequence)
    assert isinstance(target_sequence, midi.MIDIUnitSequence)
    source_len, target_len = len(source_sequence), len(target_sequence)
    cost_matrix = np.zeros([source_len, target_len], dtype=np.float32)

    for i, s in enumerate(source_sequence):
        for j, t in enumerate(target_sequence):
            if i == 0 and j == 0:
                pass
            elif i == 0:
                cost_matrix[0, j] = j
            elif j == 0:
                cost_matrix[i, 0] = i
            else:
                cost_matrix[i, j] = min(
                    [
                        cost_matrix[i - 1, j] + 1,
                        cost_matrix[i, j - 1] + 1,
                        cost_matrix[i - 1, j - 1] + cost_metric(s, t),
                    ]
                )

    return (
        cost_matrix[source_len - 1, target_len - 1] / (source_len * target_len) ** 0.5
    )


def dtw(
    source_sequence: midi.MIDIUnitSequence,
    target_sequence: midi.MIDIUnitSequence,
    cost_metric: function.DistanceMetric = function.get_distance_metric(),
) -> float:
    assert isinstance(source_sequence, midi.MIDIUnitSequence)
    assert isinstance(target_sequence, midi.MIDIUnitSequence)
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
            cost_matrix[i, j] = cost_metric(s, t) + cost

    return (
        cost_matrix[source_len - 1, target_len - 1] / (source_len * target_len) ** 0.5
    )
