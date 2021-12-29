# -*- coding: utf-8 -*-

from dataset_annotation import Annotation

import math
from midi_parser import MIDIParser
import numpy as np

from os import PathLike
from pathlib import Path

import random
from typing import Dict, Generator, Iterable, Tuple


def clip(val: float, min_=float("-inf"), max_=float("inf")) -> float:
    # val: (-inf, inf) -> [min_, max_]
    return max(min(val, max_), min_)


def load_dataset_info(
    root: PathLike, shuffle: bool = False
) -> Generator[Tuple[Path, Iterable[str]], None, None]:
    perf_dict: Dict[Path, Iterable[str]] = {}
    for score_path in Path(root).rglob("**/score.mid"):
        perf_root = score_path.parent
        perf_files = [f.name for f in perf_root.glob("*.mid") if f.stem != "score"]
        if perf_files:
            perf_dict[perf_root] = perf_files

    perf_roots = list(perf_dict.keys())
    if shuffle:
        random.shuffle(perf_roots)

    for perf_root in perf_roots:
        perf_files = perf_dict[perf_root]
        if shuffle:
            random.shuffle(perf_files)
        yield perf_root, perf_files


def new(
    root: PathLike,
    slice_duration: float = 1.0,
    expansion_rate: float = 1.5,
    fps: int = 20,
    shuffle: bool = True,
) -> Generator[Tuple[np.ndarray, np.ndarray, Tuple[int, int]], None, None]:
    midi_parser = MIDIParser()
    for perf_root, perf_files in load_dataset_info(root=root, shuffle=shuffle):
        score_midi = perf_root / "score.mid"
        score_matrix = midi_parser.process(str(score_midi), fps=fps)
        score_annotation = Annotation(path=perf_root, prefix="score")

        for perf_file in perf_files:
            perf_midi = perf_root / perf_file
            perf_matrix = midi_parser.process(str(perf_midi), fps=fps)
            perf_annotation = Annotation(path=perf_root, prefix=perf_midi.stem)

            prev_index = 0
            prev_score_onset = perf_annotation[0]
            for curr_index, curr_score_onset in enumerate(score_annotation):
                if curr_score_onset - prev_score_onset >= slice_duration:
                    score_head = math.floor(fps * prev_score_onset)
                    score_tail = math.floor(fps * curr_score_onset)
                    sliced_score_matrix: np.ndarray = np.copy(
                        score_matrix[:, score_head:score_tail]
                    )

                    prev_perf_onset = perf_annotation[prev_index]
                    curr_perf_onset = perf_annotation[curr_index]
                    perf_head = math.floor(fps * prev_perf_onset)
                    perf_tail = math.floor(fps * curr_perf_onset)
                    perf_size = perf_tail - perf_head

                    num_perf_frames = perf_matrix.shape[-1]
                    expanded_perf_size = expansion_rate * perf_size
                    expanded_perf_head = perf_head - random.randint(
                        0, math.floor((expansion_rate - 1.0) * perf_size)
                    )
                    expanded_perf_head = clip(expanded_perf_head, 0, num_perf_frames)
                    expanded_perf_head = int(expanded_perf_head)
                    expanded_perf_tail = perf_head + expanded_perf_size
                    expanded_perf_tail = clip(expanded_perf_tail, 0, num_perf_frames)
                    expanded_perf_tail = int(expanded_perf_tail)

                    sliced_perf_matrix: np.ndarray = np.copy(
                        perf_matrix[:, expanded_perf_head:expanded_perf_tail]
                    )

                    alignment = (
                        perf_head - expanded_perf_head,
                        perf_tail - expanded_perf_head,
                    )
                    yield sliced_score_matrix, sliced_perf_matrix, alignment

                    prev_index = curr_index
                    prev_score_onset = curr_score_onset


if __name__ == "__main__":
    generator = new(root="../dataset/newbie-dataset/", shuffle=True)
    score, perf, (head, tail) = next(generator)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(16, 9))
    plt.subplot(2, 1, 1)
    plt.title("Score")
    plt.imshow(
        score, cmap="gray", aspect="auto", origin="lower", interpolation="nearest"
    )
    plt.ylabel("MIDI Key")
    plt.xlabel("Frame")
    plt.ylim([0, 128])
    plt.subplot(2, 1, 2)
    plt.title("Performance")
    plt.imshow(
        perf, cmap="gray", aspect="auto", origin="lower", interpolation="nearest"
    )
    plt.ylabel("MIDI Key")
    plt.xlabel("Frame")
    plt.vlines(head, 0, 128, label="head")
    plt.vlines(tail, 0, 128, label="tail")
    plt.legend()
    plt.ylim([0, 128])
    plt.show()
