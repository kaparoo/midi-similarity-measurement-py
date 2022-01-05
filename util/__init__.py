# -*- coding: utf-8 -*-


try:
    from metric import score_alignment
    from visualization import plot_midi_matrices
except ImportError:
    from .metric import score_alignment
    from .visualization import plot_midi_matrices


__all__ = ["plot_midi_matrices", "score_alignment"]
