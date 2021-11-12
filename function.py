# -*- coding: utf-8 -*-

import midi
from typing import Callable

# Decay functions for converting midi matrix to MIDIUnitSequenceList
DecayFn = Callable[[float], float]


def exponential_decay_factory(settling_frame: int) -> DecayFn:
    geometric_ratio = 0.1 ** (1 / settling_frame)

    def exponential_decay(prev_velocity: float) -> float:
        assert prev_velocity > 0
        return prev_velocity * geometric_ratio

    return exponential_decay


# Metrics for choosing significant unit for each `MIDIUnitSequence`
UnitMetric = Callable[[midi.MIDIUnitSequence], midi.MIDIUnit]


def onset_nearest_unit_factory(
    settling_frame: int, compensation_frame: int
) -> UnitMetric:
    geometric_ratio = 0.1 ** (1 / settling_frame)
    unbalanced_comp_weight = (1 / geometric_ratio) ** compensation_frame

    def onset_nearest_unit(sequence: midi.MIDIUnitSequence) -> midi.MIDIUnit:
        significant_unit = sequence[0]
        for unit in sequence[1:]:
            if (
                significant_unit.get_velocity()
                <= unbalanced_comp_weight * unit.get_velocity()
            ):
                significant_unit = unit
        return significant_unit

    return onset_nearest_unit


# Cost metrics for time warping algorithms
CostMetric = Callable[[midi.MIDIUnit, midi.MIDIUnit], float]


def compare_cost_fn(s: midi.MIDIUnit, t: midi.MIDIUnit) -> float:
    return float(s.get_midi_key() != t.get_midi_key())


def distance_cost_fn(s: midi.MIDIUnit, t: midi.MIDIUnit) -> float:
    if s.is_note() and t.is_note():
        k1, v1 = s.get_values()
        k2, v2 = t.get_values()
        return ((k1 - k2) ** 2 + (v1 - v2) ** 2) ** 0.5
    elif s.is_note():
        k, v = s.get_values()
        return k + v
    elif t.is_note():
        k, v = t.get_values()
        return k + v
    else:
        return 0
