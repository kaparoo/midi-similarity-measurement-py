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
    from measurement import measure, Similarity
except ImportError:
    from .algorithm import (
        CostFn,
        compare_midi_key,
        dtw,
        euclidean,
        global_dtw,
        subsequence_dtw,
    )
    from .measurement import measure, Similarity


__all__ = [
    "CostFn",
    "compare_midi_key",
    "dtw",
    "euclidean",
    "global_dtw",
    "subsequence_dtw",
    "measure",
    "Similarity",
]
