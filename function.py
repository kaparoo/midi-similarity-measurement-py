# -*- coding: utf-8 -*-

import midi
from typing import Any, Callable

# Decay functions for converting midi matrix to MIDIUnitSequenceList
DecayFn = Callable[[float], float]


def reversed_ramp_decay(prev_velocity: float) -> float:
    assert prev_velocity >= 1
    if prev_velocity > 1:
        return prev_velocity - 1
    else:
        return 1


def exponential_decay_factory(settling_frame: int) -> DecayFn:
    geometric_ratio = 0.1 ** (1 / settling_frame)

    def exponential_decay(prev_velocity: float) -> float:
        assert prev_velocity > 0
        return prev_velocity * geometric_ratio

    return exponential_decay


ALLOWED_DECAY_FNS = ["reversed_ramp", "exponential"]


def get_decay_fn(fn_name: str = "exponential", *args) -> DecayFn:
    if fn_name == "reversed_ramp":
        return reversed_ramp_decay
    else:
        return exponential_decay_factory(*args)


# Metrics for choosing significant unit for each `MIDIUnitSequence`
UnitMetric = Callable[[midi.MIDIUnitSequence], midi.MIDIUnit]


def max_midi_key_unit(sequence: midi.MIDIUnitSequence) -> midi.MIDIUnit:
    return sequence[-1]


def onset_nearest_unit_factory(
    settling_frame: int, compensate_frame: int
) -> UnitMetric:
    geometric_ratio = 0.1 ** (1 / settling_frame)
    unbalanced_comp_weight = 1 / geometric_ratio ** compensate_frame

    def onset_nearest_unit(sequence: midi.MIDIUnitSequence) -> midi.MIDIUnit:
        significant_unit = sequence[0]
        for i in range(1, len(sequence)):
            if (
                significant_unit.velocity
                <= unbalanced_comp_weight * sequence[i].velocity
            ):
                significant_unit = sequence[i]
        return significant_unit

    return onset_nearest_unit


ALLOWED_UNIT_METRICS = ["max_midi_key", "onset_nearest"]


def get_unit_metric(metric_name: str = "max_midi_key", *args) -> UnitMetric:
    if metric_name == "onset_nearest":
        return onset_nearest_unit_factory(*args)
    else:
        return max_midi_key_unit


# Distance metrics for time warping algorithms
DistanceMetric = Callable[[midi.MIDIUnit, midi.MIDIUnit], float]


def default_distance(s: midi.MIDIUnit, t: midi.MIDIUnit) -> float:
    # A cosine similarity is supported by substracting two `MIDIUnit` objects
    # See MIDIUnit.__sub__ (midi.py/20)
    return abs(s - t)


ALLOWED_DISTANCE_METRICS = ["default"]


def get_distance_metric(metric_name: str = "default") -> DistanceMetric:
    return default_distance
