# -*- coding: utf-8 -*-

try:
    from algorithm import (
        CostFn,
        compare_midi_key,
        dtw,
        euclidean,
        global_dtw,
        subsequence_dtw,
    )
    from measurement import measure
except ImportError:
    from .algorithm import (
        CostFn,
        compare_midi_key,
        dtw,
        euclidean,
        global_dtw,
        subsequence_dtw,
    )
    from .measurement import measure


__all__ = [
    "CostFn",
    "compare_midi_key",
    "dtw",
    "euclidean",
    "global_dtw",
    "subsequence_dtw",
    "measure",
]
