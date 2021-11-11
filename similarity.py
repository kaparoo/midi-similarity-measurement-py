# -*- coding: utf-8 -*-

import algorithm
import function
import midi
import numpy as np
from typing import Tuple


def score_similarity(
    source_midi_matrix: np.ndarray,
    target_midi_matrix: np.ndarray,
    onset_weight: int = 10.0,
    decay_fn: function.DecayFn = function.reversed_ramp_decay,
    unit_metric: function.UnitMetric = function.max_midi_key_unit,
    distance_metric: function.DistanceMetric = lambda s, t: float(
        s.get_midi_key() != t.get_midi_key()
    ),
    time_warping_algorithm: str = "levenshtein",
) -> Tuple[float, float]:

    source = midi.MIDIUnitSequenceList.from_midi_matrix(
        source_midi_matrix, onset_weight, decay_fn
    )
    target = midi.MIDIUnitSequenceList.from_midi_matrix(
        target_midi_matrix, onset_weight, decay_fn
    )

    source_histogram = source.to_pitch_histogram()
    target_histogram = target.to_pitch_histogram()

    source_sequence = source.to_significant_unit_sequence(unit_metric)
    target_sequence = target.to_significant_unit_sequence(unit_metric)

    pitch_similarity = algorithm.euclidean(source_histogram, target_histogram)

    if time_warping_algorithm == "dtw":
        significant_note_similarity = algorithm.dtw(
            source_sequence, target_sequence, distance_metric,
        )
    else:
        significant_note_similarity = algorithm.levenshtein(
            source_sequence, target_sequence, distance_metric,
        )

    return pitch_similarity, significant_note_similarity


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
