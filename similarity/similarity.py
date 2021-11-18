# -*- coding: utf-8 -*-

import midi
import numbers
import numpy as np
import time
from typing import Dict, Tuple
from . import algorithm

# TODO(kaparoo): Rewrite error messages more properly
def _verify_arguments(
    source_midi_matrix: np.ndarray,
    target_midi_matrix: np.ndarray,
    settling_frame: numbers.Integral,
    compensation_frame: numbers.Integral,
) -> Tuple[np.ndarray, np.ndarray, int, int]:

    if not isinstance(source_midi_matrix, np.ndarray):
        raise TypeError(type(source_midi_matrix))
    else:
        shape = source_midi_matrix.shape
        if shape[0] != midi.NUM_MIDI_KEYS:
            raise ValueError(shape)
        elif len(shape) > 2:
            return ValueError(shape)
        elif len(shape) == 1:
            source_midi_matrix = np.reshape(source_midi_matrix, [midi.NUM_MIDI_KEYS, 1])

    if not isinstance(target_midi_matrix, np.ndarray):
        raise TypeError(type(target_midi_matrix))
    else:
        shape = target_midi_matrix.shape
        if shape[0] != midi.NUM_MIDI_KEYS:
            raise ValueError(shape)
        elif len(shape) > 2:
            return ValueError(shape)
        elif len(shape) == 1:
            target_midi_matrix = np.reshape(target_midi_matrix, [midi.NUM_MIDI_KEYS, 1])

    if not isinstance(settling_frame, numbers.Integral):
        raise TypeError(type(settling_frame))
    elif settling_frame <= 0:
        raise ValueError(settling_frame)
    else:
        settling_frame = int(settling_frame)

    if not isinstance(compensation_frame, numbers.Integral):
        raise TypeError(type(compensation_frame))
    elif compensation_frame < 0:
        raise ValueError(compensation_frame)
    else:
        compensation_frame = int(compensation_frame)

    return source_midi_matrix, target_midi_matrix, settling_frame, compensation_frame


def measure(
    source_midi_matrix: np.ndarray,
    target_midi_matrix: np.ndarray,
    settling_frame: int = 10,
    compensation_frame: int = 0,
    use_subsequence_dtw: bool = True,
    use_decay_for_histogram: bool = True,
    cost_metric: algorithm.CostMetric = algorithm.compare_cost_fn,
    return_execution_times: bool = False,
) -> Tuple[float, float, Dict[str, float]]:

    (
        source_midi_matrix,
        target_midi_matrix,
        settling_frame,
        compensation_frame,
    ) = _verify_arguments(
        source_midi_matrix, target_midi_matrix, settling_frame, compensation_frame
    )

    if return_execution_times:
        timestamp1 = time.time()

    source = midi.MIDIUnitSequenceList.from_midi_matrix(
        source_midi_matrix, settling_frame
    )
    target = midi.MIDIUnitSequenceList.from_midi_matrix(
        target_midi_matrix, settling_frame
    )

    if return_execution_times:
        timestamp2 = time.time()

    source_sequence = source.to_representative_unit_sequence(compensation_frame)
    target_sequence = target.to_representative_unit_sequence(compensation_frame)

    if return_execution_times:
        timestamp3 = time.time()

    if use_subsequence_dtw:
        timewarping_distance, (head, tail), _ = algorithm.subsequence_dtw(
            source_sequence, target_sequence, cost_metric, stabilize=True
        )
        if return_execution_times:
            timestamp4 = time.time()

        if use_decay_for_histogram:
            source_histogram = source.to_pitch_histogram(with_decay=True)
            target_histogram = target[head : tail + 1].to_pitch_histogram(
                with_decay=True
            )
        else:
            source_histogram = source.to_pitch_histogram(with_decay=False)
            target_histogram = target[head : tail + 1].to_pitch_histogram(
                with_decay=False
            )

    else:
        timewarping_distance = algorithm.levenshtein(
            source_sequence, target_sequence, cost_metric, stabilize=True
        )
        if return_execution_times:
            timestamp4 = time.time()

        if use_decay_for_histogram:
            source_histogram = source.to_pitch_histogram(with_decay=True)
            target_histogram = target.to_pitch_histogram(with_decay=True)
        else:
            source_histogram = source.to_pitch_histogram(with_decay=False)
            target_histogram = target.to_pitch_histogram(with_decay=False)

    if return_execution_times:
        timestamp5 = time.time()

    histogram_distance = algorithm.euclidean(source_histogram, target_histogram)

    if return_execution_times:
        timestamp6 = time.time()

    if not return_execution_times:
        execution_times = {}
    else:
        execution_times = {
            "from_midi_matrix": timestamp2 - timestamp1,
            "to_representative_unit_sequence": timestamp3 - timestamp2,
            "timewarping": timestamp4 - timestamp3,
            "to_pitch_histogram": timestamp5 - timestamp4,
            "euclidean": timestamp6 - timestamp5,
            "total": timestamp6 - timestamp1,
        }

    return (
        histogram_distance,
        timewarping_distance,
        execution_times,
    )
