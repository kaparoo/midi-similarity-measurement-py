# -*- coding: utf-8 -*-

import algorithm
import function
import midi
import numbers
import numpy as np
from typing import Dict, Tuple
from time import time


# TODO(kaparoo): Rewrite error messages more properly
def _verify_arguments(
    source_midi_matrix: np.ndarray, target_midi_matrix: np.ndarray, onset_weight: float,
) -> Tuple[np.ndarray, np.ndarray, float]:

    if not isinstance(source_midi_matrix, np.ndarray):
        raise TypeError(type(source_midi_matrix))
    else:
        shape_ = source_midi_matrix.shape
        if shape_[0] != midi.NUM_MIDI_KEYS:
            raise ValueError(shape_)
        elif len(shape_) > 2:
            return ValueError(shape_)
        elif len(shape_) == 1:
            source_midi_matrix = np.reshape(source_midi_matrix, [midi.NUM_MIDI_KEYS, 1])

    if not isinstance(target_midi_matrix, np.ndarray):
        raise TypeError(type(target_midi_matrix))
    else:
        shape_ = target_midi_matrix.shape
        if shape_[0] != midi.NUM_MIDI_KEYS:
            raise ValueError(shape_)
        elif len(shape_) > 2:
            return ValueError(shape_)
        elif len(shape_) == 1:
            target_midi_matrix = np.reshape(target_midi_matrix, [midi.NUM_MIDI_KEYS, 1])

    if not isinstance(onset_weight, numbers.Real):
        raise TypeError(type(onset_weight))
    elif onset_weight <= 0.0:
        raise ValueError(onset_weight)
    else:
        onset_weight = float(onset_weight)

    return source_midi_matrix, target_midi_matrix, onset_weight


def score_similarity(
    source_midi_matrix: np.ndarray,
    target_midi_matrix: np.ndarray,
    onset_weight: float,
    decay_fn: function.DecayFn,
    unit_metric: function.UnitMetric,
    cost_metric: function.CostMetric,
    check_execution_times: bool = False,
) -> Tuple[float, float, Dict[str, float]]:

    source_midi_matrix, target_midi_matrix, onset_weight = _verify_arguments(
        source_midi_matrix, target_midi_matrix, onset_weight
    )

    if check_execution_times:
        timestamp1 = time()

    source = midi.MIDIUnitSequenceList.from_midi_matrix(
        source_midi_matrix, onset_weight, decay_fn
    )
    target = midi.MIDIUnitSequenceList.from_midi_matrix(
        target_midi_matrix, onset_weight, decay_fn
    )

    if check_execution_times:
        timestamp2 = time()

    source_histogram = source.to_pitch_histogram()
    target_histogram = target.to_pitch_histogram()

    if check_execution_times:
        timestamp3 = time()

    source_sequence = source.to_significant_unit_sequence(unit_metric)
    target_sequence = target.to_significant_unit_sequence(unit_metric)

    if check_execution_times:
        timestamp4 = time()

    pitch_similarity = algorithm.euclidean(source_histogram, target_histogram)

    if check_execution_times:
        timestamp5 = time()

    significant_note_similarity = algorithm.levenshtein(
        source_sequence, target_sequence, cost_metric,
    )

    if check_execution_times:
        timestamp6 = time()

    if not check_execution_times:
        return pitch_similarity, significant_note_similarity, {}
    else:
        execution_times = {
            "from_midi_matrix": timestamp2 - timestamp1,
            "to_pitch_histogram": timestamp3 - timestamp2,
            "to_significant_unit_sequence": timestamp4 - timestamp3,
            "euclidean": timestamp5 - timestamp4,
            "timewarping": timestamp6 - timestamp5,
            "total": timestamp6 - timestamp1,
        }
        return pitch_similarity, significant_note_similarity, execution_times


if __name__ == "__main__":
    source = [[0, 9, 3], [1, 3, 5, 4, 2], [9], [37, 11]]
    target = [[3, 127, 91, 1, 0], [11, 22, 33, 44, 55, 66, 77]]

    source_matrix = np.zeros([midi.NUM_MIDI_KEYS, len(source)], dtype=np.float32)
    for frame_idx, pressed_keys in enumerate(source):
        source_matrix[pressed_keys, frame_idx] = 1
    source_midi_seq_list = midi.MIDIUnitSequenceList.from_midi_matrix(
        source_matrix, onset_weight=1.0
    )
    # source: [[(0, 1.0), (9, 1.0), (3, 1.0)], [(1, 1.0), (3, 1.0), (5, 1.0), (4, 1.0), (2, 1.0)],
    #          [(9, 1.0)], [(37, 1.0), (11, 1.0)]]
    print(f"source: {source_midi_seq_list}")

    target_matrix = np.zeros([midi.NUM_MIDI_KEYS, len(target)], dtype=np.float32)
    for frame_idx, pressed_keys in enumerate(target):
        target_matrix[pressed_keys, frame_idx] = 1
    target_midi_seq_list = midi.MIDIUnitSequenceList.from_midi_matrix(
        target_matrix, onset_weight=1.0
    )
    # target: [[(3, 1.0), (127, 1.0), (91, 1.0), (1, 1.0), (0, 1.0)],
    #          [(11, 1.0), (22, 1.0), (33, 1.0), (44, 1.0), (55, 1.0), (66, 1.0), (77, 1.0)]]
    print(f"target: {target_midi_seq_list}")

    source_histogram = source_midi_seq_list.to_pitch_histogram()
    target_histogram = target_midi_seq_list.to_pitch_histogram()
    print(f"Pitch histogram:")
    # source: [0.09090909 0.18181819 0.09090909 0.18181819 0.09090909 0.09090909
    #          0.         0.         0.         0.18181819 0.         0.09090909]
    print(f"\tsource: {source_histogram}")
    # target: [0.08333334 0.08333334 0.         0.08333334 0.         0.08333334
    #          0.08333334 0.25       0.08333334 0.08333334 0.08333334 0.08333334]
    print(f"\ttarget: {target_histogram}")

    source_sequence = source_midi_seq_list.to_significant_unit_sequence()
    target_sequence = target_midi_seq_list.to_significant_unit_sequence()
    print(f"Significant unit sequence:")
    # source: [(9, 1.0), (5, 1.0), (9, 1.0), (37, 1.0)]
    print(f"\tsource: {source_sequence}")
    # target: [(127, 1.0), (77, 1.0)]
    print(f"\ttarget: {target_sequence}")

    euclidean_similarity = algorithm.euclidean(source_histogram, target_histogram)
    levenshtein_similarity = algorithm.levenshtein(
        source_sequence,
        target_sequence,
        lambda s, t: float(s.get_midi_key() != t.get_midi_key()),
    )

    # Euclidean similarity: 0.35934976820895137
    print(f"Euclidean similarity: {euclidean_similarity}")
    # Levenshtein similarity: 3.0
    print(f"Levenshtein similarity: {levenshtein_similarity}")
