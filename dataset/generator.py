# -*- coding: utf-8 -*-

import math
import numpy as np
import numpy.random as random
from os import PathLike
from pathlib import Path
from typing import Generator, List, Sequence, Tuple, Union

try:
    from annotation import Annotation
    from midi_parser import MIDIParser
except ImportError:
    from .annotation import Annotation
    from .midi_parser import MIDIParser


__all__ = ["Dataset", "new_generator"]


def _clip(val: float, min_=float("-inf"), max_=float("inf")) -> float:
    # val: (-inf, inf) -> [min_, max_]
    return max(min(val, max_), min_)


def _load_dataset_info(
    root: PathLike, score: str = "score", shuffle: bool = False
) -> Generator[Tuple[Path, Sequence[str]], None, None]:
    perf_roots: List[Path] = []
    for score_path in Path(root).rglob(f"**/{score}.mid"):
        perf_roots.append(score_path.parent)
    if shuffle:
        random.shuffle(perf_roots)
    for perf_root in perf_roots:
        perf_files = [f.name for f in perf_root.glob("*.mid") if f.stem != score]
        if perf_files:
            if shuffle:
                random.shuffle(perf_files)
            yield perf_root, perf_files


Dataset = Tuple[np.ndarray, np.ndarray, Tuple[int, int]]


def new_generator(
    root: PathLike,
    score_prefix: str = "score",
    slice_duration: Union[float, Tuple[float, float]] = 1.0,
    expansion_rate: Union[float, Tuple[float, float]] = 1.5,
    frames_per_second: int = 20,
    mark_onset: bool = True,
    shuffle: bool = True,
    verbose: bool = False,
) -> Generator[Union[Dataset, Tuple[Dataset, Tuple[Path, float, float]]], None, None,]:
    midi_parser = MIDIParser()

    get_slice_duration = lambda: slice_duration
    if not isinstance(slice_duration, float):
        start1, end1 = slice_duration
        get_slice_duration = lambda: random.uniform(start1, end1)

    get_expansion_rate = lambda: expansion_rate
    if not isinstance(expansion_rate, float):
        start2, end2 = expansion_rate
        get_expansion_rate = lambda: random.uniform(start2, end2)

    for perf_root, perf_files in _load_dataset_info(root, score_prefix, shuffle):
        score_midi = perf_root / f"{score_prefix}.mid"
        _, score_matrix = midi_parser(score_midi, frames_per_second, mark_onset)
        score_annotation = Annotation(path=perf_root, prefix=score_prefix)

        for perf_file in perf_files:
            perf_midi = perf_root / perf_file
            _, perf_matrix = midi_parser(perf_midi, frames_per_second, mark_onset)
            perf_annotation = Annotation(path=perf_root, prefix=perf_midi.stem)
            num_prev_annotations = len(perf_annotation)

            prev_index = 0
            prev_score_onset = perf_annotation[0]
            for curr_index, curr_score_onset in enumerate(score_annotation):
                if curr_index >= num_prev_annotations:
                    break

                slice_duration = get_slice_duration()
                expansion_rate = get_expansion_rate()

                if curr_score_onset - prev_score_onset >= slice_duration:
                    score_head = math.floor(frames_per_second * prev_score_onset)
                    score_tail = math.floor(frames_per_second * curr_score_onset)
                    score_slice: np.ndarray = np.copy(
                        score_matrix[:, score_head:score_tail]
                    )

                    prev_perf_onset = perf_annotation[prev_index]
                    curr_perf_onset = perf_annotation[curr_index]
                    perf_head = math.floor(frames_per_second * prev_perf_onset)
                    perf_tail = math.floor(frames_per_second * curr_perf_onset)
                    perf_size = perf_tail - perf_head

                    num_perf_frames = perf_matrix.shape[-1]
                    expanded_perf_size = expansion_rate * perf_size
                    expanded_perf_head = perf_head - random.randint(
                        0, math.floor((expansion_rate - 1.0) * perf_size)
                    )
                    expanded_perf_head = _clip(expanded_perf_head, 0, num_perf_frames)
                    expanded_perf_head = int(expanded_perf_head)
                    expanded_perf_tail = perf_head + expanded_perf_size
                    expanded_perf_tail = _clip(expanded_perf_tail, 0, num_perf_frames)
                    expanded_perf_tail = int(expanded_perf_tail)

                    perf_slice: np.ndarray = np.copy(
                        perf_matrix[:, expanded_perf_head:expanded_perf_tail]
                    )

                    alignment = (
                        perf_head - expanded_perf_head,
                        perf_tail - expanded_perf_head,
                    )

                    if not verbose:
                        yield score_slice, perf_slice, alignment
                    else:
                        yield (score_slice, perf_slice, alignment), (
                            perf_midi,
                            slice_duration,
                            expansion_rate,
                        )

                    prev_index = curr_index
                    prev_score_onset = curr_score_onset


if __name__ == "__main__":
    generator = new_generator(root="../dataset/newbie-dataset/", shuffle=True)
    score, perf, (head, tail) = next(generator)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(16, 9), facecolor="white")
    plt.subplot(2, 1, 1)
    plt.title("Score")
    plt.imshow(
        score, cmap="gray", aspect="auto", origin="lower", interpolation="nearest"
    )
    plt.ylabel("MIDI Key")
    plt.xlabel("Frame")
    xmax = score.shape[-1] - 1
    plt.xlim([0, xmax])
    plt.ylim([0, 127])
    plt.subplot(2, 1, 2)
    plt.title("Performance")
    plt.imshow(
        perf, cmap="gray", aspect="auto", origin="lower", interpolation="nearest"
    )
    plt.ylabel("MIDI Key")
    plt.xlabel("Frame")
    plt.vlines(head, 0, 127, "r", label="head")
    plt.vlines(tail, 0, 127, "r", label="tail")
    plt.legend()
    xmax = perf.shape[-1] - 1
    plt.xlim([0, xmax])
    plt.ylim([0, 127])
    plt.tight_layout()
    plt.show()
