# -*- coding: utf-8 -*-


try:
    from visualization import plot_midi_matrices
except ImportError:
    from .visualization import plot_midi_matrices


__all__ = ["plot_midi_matrices"]
