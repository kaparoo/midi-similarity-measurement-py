import matplotlib.pyplot as plt
import midi
import numpy as np
import pathlib
from typing import Union


def plot_midi_matrices(
    score: np.ndarray,
    perf: np.ndarray,
    head: int = None,
    tail: int = None,
    title: str = None,
) -> None:
    plt.figure(figsize=(16, 9))
    if title:
        plt.suptitle(title)
    plt.subplot(2, 1, 1)
    plt.title("Score")
    plt.imshow(
        score, cmap="gray", aspect="auto", origin="lower", interpolation="nearest"
    )
    plt.ylabel("MIDI Key")
    plt.xlabel("Frame")
    plt.ylim(midi.MIN_MIDI_KEY, midi.MAX_MIDI_KEY)

    plt.subplot(2, 1, 2)
    plt.title("Performance")
    plt.imshow(
        perf, cmap="gray", aspect="auto", origin="lower", interpolation="nearest"
    )
    plt.plot([head, head], [midi.MIN_MIDI_KEY, midi.MAX_MIDI_KEY])
    plt.plot([tail, tail], [midi.MIN_MIDI_KEY, midi.MAX_MIDI_KEY])
    plt.ylabel("MIDI Key")
    plt.xlabel("Frame")
    plt.ylim(midi.MIN_MIDI_KEY, midi.MAX_MIDI_KEY)
    plt.show()
    plt.clf()


def process_decay_to_midi_matrix(
    midi_matrix: np.ndarray, settling_frame: int = 8
) -> np.ndarray:
    prev_pressed = [False] * midi.NUM_MIDI_KEYS
    midi_matrix = np.reshape(midi_matrix.copy(), [midi.NUM_MIDI_KEYS, -1]).T
    for frame_idx in range(len(midi_matrix)):
        for midi_key in range(midi.NUM_MIDI_KEYS):
            velocity = 0
            if midi_matrix[frame_idx, midi_key] <= 0:
                prev_pressed[midi_key] = False
            elif frame_idx > 0:
                prev_velocity = midi_matrix[frame_idx - 1, midi_key]
                if prev_velocity > 0:
                    velocity = prev_velocity - 1
                elif not prev_pressed[midi_key]:
                    velocity = settling_frame - 1
                prev_pressed[midi_key] = True
            else:
                velocity = settling_frame - 1
                prev_pressed[midi_key] = True
            midi_matrix[frame_idx, midi_key] = velocity
    return midi_matrix.T


def plot_scatters(
    save_root: Union[str, pathlib.Path],
    pos_similarities: np.ndarray,
    neg_similarities: np.ndarray,
):
    if isinstance(save_root, str):
        save_root = pathlib.Path(save_root)

    fig = plt.figure("scatter_2d")
    ax = fig.gca()
    ax.set_xlabel("Histogram distance")
    ax.set_ylabel("Timewarping distance")
    ax.scatter(pos_similarities[:, 0], pos_similarities[:, 1], c="k", label="Positive")
    ax.scatter(
        neg_similarities[:, 0],
        neg_similarities[:, 1],
        c="w",
        edgecolors="k",
        label="Negative",
    )
    plt.legend()
    plt.savefig(save_root / "scatter_2d.png")
    plt.clf()

    fig = plt.figure("scatter_3d")
    ax = fig.gca(projection="3d")
    ax.set_xlabel("Histogram distance")
    ax.set_ylabel("Timewarping distance")
    ax.set_zlabel("Length ratio")
    ax.scatter(
        pos_similarities[:, 0],
        pos_similarities[:, 1],
        pos_similarities[:, 2],
        c="k",
        label="Positive",
    )
    ax.scatter(
        neg_similarities[:, 0],
        neg_similarities[:, 1],
        neg_similarities[:, 2],
        c="w",
        edgecolors="k",
        label="Negative",
    )
    plt.legend()
    plt.savefig(save_root / "scatter_3d.png")
    plt.clf()
