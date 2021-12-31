# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple


__all__ = ["plot_midi_matrix"]


_MIDI_MATRIX_IMSHOW_OPTION = {
    "cmap": "gray",
    "aspect": "auto",
    "origin": "lower",
    "interpolation": "nearest",
}


def plot_midi_matrices(
    matrix1: np.ndarray,
    matrix2: np.ndarray,
    alignment: Optional[Tuple[int, int]] = None,
    title1: str = "Score",
    title2: str = "Performance",
) -> None:
    plt.figure(figsize=(16, 9), facecolor="white")

    plt.subplot(2, 1, 1)
    if title1 is not None:
        plt.title(title1)
    plt.imshow(matrix1, **_MIDI_MATRIX_IMSHOW_OPTION)
    plt.ylabel("MIDI Key")
    plt.xlabel("Frame")
    ymax, xmax = matrix1.shape
    plt.ylim([0, ymax - 1])
    plt.xlim([0, xmax - 1])

    plt.subplot(2, 1, 2)
    if title2 is not None:
        plt.title(title2)
    plt.imshow(matrix2, **_MIDI_MATRIX_IMSHOW_OPTION)
    plt.ylabel("MIDI Key")
    plt.xlabel("Frame")
    ymax, xmax = matrix2.shape
    if alignment is not None:
        head, tail = alignment
        plt.vlines(head, 0, ymax - 1, "r", label="head")
        plt.vlines(tail, 0, ymax - 1, "r", label="tail")
        plt.legend()
    plt.ylim([0, ymax - 1])
    plt.xlim([0, xmax - 1])

    plt.tight_layout()
    plt.show()
