# -*- coding: utf-8 -*-

import math
import midi
import numbers
import numpy as np
import os
import pathlib
import random
from typing import Any, Generator, List, Tuple, Union
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="madmom")

__all__ = ["spawn"]

_SCORE_MIDI_FILE = "midi_score.mid"


def _clamp(val: numbers.Real, min_=float("-inf"), max_=float("inf")):
    return max(min(val, max_), min_)


def _debug_log(*args: Any):
    if __name__ == "__main__":
        print(*args)


# TODO(kaparoo): Rewrite error messages more properly
def _verify_arguments(
    dataset_root: Union[pathlib.Path, str],
    slice_duration: numbers.Real,
    expansion_rate: numbers.Real,
    frame_per_second: numbers.Integral,
) -> Tuple[pathlib.Path, float, float, int]:

    if not isinstance(dataset_root, (pathlib.Path, str)):
        raise TypeError(type(dataset_root))
    elif isinstance(dataset_root, str):
        dataset_root = dataset_root.strip()
        if not dataset_root:
            raise ValueError()  # empty string
        else:
            dataset_root = pathlib.Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)
    elif not dataset_root.is_dir():
        raise FileExistsError(dataset_root)

    if not isinstance(slice_duration, numbers.Real):
        raise TypeError(type(slice_duration))
    elif slice_duration <= 0.0:
        raise ValueError(slice_duration)
    else:
        slice_duration = float(slice_duration)

    if not isinstance(expansion_rate, numbers.Real):
        raise TypeError(type(expansion_rate))
    elif expansion_rate <= 1.0:
        raise ValueError(expansion_rate)
    else:
        expansion_rate = float(expansion_rate)

    if not isinstance(frame_per_second, numbers.Integral):
        raise TypeError(type(frame_per_second))
    elif frame_per_second <= 0:
        raise ValueError(frame_per_second)
    else:
        frame_per_second = float(frame_per_second)

    return dataset_root, slice_duration, expansion_rate, frame_per_second


def spawn(
    dataset_root: Union[pathlib.Path, str],
    slice_duration: numbers.Real = 1.0,
    expansion_rate: numbers.Real = 1.5,
    frame_per_second: int = 20,
    shuffle: bool = False,
) -> Generator[Tuple[np.ndarray, np.ndarray, Tuple[int, int]], None, None]:
    _debug_log(dataset_root, slice_duration, expansion_rate, frame_per_second)
    dataset_root, slice_duration, expansion_rate, frame_per_second = _verify_arguments(
        dataset_root, slice_duration, expansion_rate, frame_per_second
    )
    _debug_log(dataset_root, slice_duration, expansion_rate, frame_per_second)

    midi_parser = midi.MIDIParser(show=False)

    # TODO(kaparoo): Replace logic to use methods of pathlib.Path (e.g. iterdir, glob)
    dataset_infos: List[Tuple[pathlib.Path, List[str]]] = []
    for root, _, files in os.walk(dataset_root):
        if _SCORE_MIDI_FILE in files:
            root = pathlib.Path(root)
            perf_files = filter(
                lambda file: file is not _SCORE_MIDI_FILE and file[-4:] == ".mid",
                files,
            )
            dataset_infos.append((root, perf_files))
    else:
        if not dataset_infos:
            raise FileNotFoundError()
        if shuffle:
            random.shuffle(dataset_infos)

    for root_path, perf_midi_files in dataset_infos:
        _debug_log(f"root_path: {root_path}")

        score_midi_path = root_path / _SCORE_MIDI_FILE
        _, _, _, score_midi_matrix, _ = midi_parser.process(
            str(score_midi_path), return_midi_matrix=True, fps=frame_per_second
        )  # score_midi_matrix: 128 x N'
        score_annotation = midi.Annotation(root_path, score_midi_path.stem)

        _debug_log(f"score_midi_path: {score_midi_path}")
        _debug_log(f"score_midi_matrix.shape: {score_midi_matrix.shape}")
        _debug_log("----------------------------------------")

        for perf_midi_file in perf_midi_files:
            perf_midi_path = root_path / perf_midi_file
            _, _, _, perf_midi_matrix, _ = midi_parser.process(
                str(perf_midi_path), return_midi_matrix=True, fps=frame_per_second
            )  # perf_midi_matrix: 128 x M'
            perf_annotation = midi.Annotation(root_path, perf_midi_path.stem)

            _debug_log(f"perf_midi_path: {perf_midi_path}")
            _debug_log(f"perf_midi_matrix.shape: {perf_midi_matrix.shape}")

            prev_index = 0
            prev_score_onset = score_annotation[0]
            for curr_index, curr_score_onset in enumerate(score_annotation):
                if curr_score_onset - prev_score_onset >= slice_duration:
                    _debug_log(f"[INDEX {curr_index}]")
                    try:
                        _debug_log(f"prev_score_onset: {prev_score_onset}")
                        _debug_log(f"curr_score_onset: {curr_score_onset}")

                        score_matrix_head = math.floor(
                            frame_per_second * prev_score_onset
                        )
                        score_matrix_tail = math.floor(
                            frame_per_second * curr_score_onset
                        )

                        _debug_log(f"score_matrix_head: {score_matrix_head}")
                        _debug_log(f"score_matrix_tail: {score_matrix_tail}")

                        sliced_score_midi_matrix: np.ndarray = score_midi_matrix[
                            :, score_matrix_head:score_matrix_tail
                        ]  # 128 x N

                        _debug_log(
                            f"sliced_score_midi_matrix.shape: {sliced_score_midi_matrix.shape}"
                        )

                        prev_perf_onset = perf_annotation[prev_index]
                        curr_perf_onset = perf_annotation[curr_index]

                        _debug_log(f"prev_perf_onset: {prev_perf_onset}")
                        _debug_log(f"curr_perf_onset: {curr_perf_onset}")

                        perf_matrix_head = math.floor(
                            frame_per_second * prev_perf_onset
                        )
                        perf_matrix_tail = math.floor(
                            frame_per_second * curr_perf_onset
                        )
                        perf_len = perf_matrix_tail - perf_matrix_head

                        _debug_log(f"perf_head: {perf_matrix_head}")
                        _debug_log(f"perf_tail: {perf_matrix_tail}")
                        _debug_log(f"perf_len: {perf_len}")

                        num_perf_frames = perf_midi_matrix.shape[-1]
                        expanded_perf_len = expansion_rate * perf_len
                        expanded_perf_matrix_head = perf_matrix_head - random.randint(
                            0, math.floor((expansion_rate - 1.0) * perf_len)
                        )
                        expanded_perf_matrix_head = int(
                            _clamp(expanded_perf_matrix_head, 0.0, num_perf_frames,)
                        )
                        expanded_perf_matrix_tail = perf_matrix_head + expanded_perf_len
                        expanded_perf_matrix_tail = int(
                            _clamp(expanded_perf_matrix_tail, 0.0, num_perf_frames,)
                        )

                        _debug_log(f"expanded_perf_head: {expanded_perf_matrix_head}")
                        _debug_log(f"expanded_perf_tail: {expanded_perf_matrix_tail}")
                        _debug_log(f"expanded_perf_len: {expanded_perf_len}")

                        sliced_perf_midi_matrix: np.ndarray = perf_midi_matrix[
                            :, expanded_perf_matrix_head:expanded_perf_matrix_tail
                        ]  # 128 x M

                        _debug_log(
                            f"sliced_perf_midi_matrix.shape: {sliced_perf_midi_matrix.shape}"
                        )

                        yield sliced_score_midi_matrix.copy(), sliced_perf_midi_matrix.copy(), (
                            perf_matrix_head - expanded_perf_matrix_head,
                            perf_matrix_tail - expanded_perf_matrix_head,
                        )
                    except Exception as e:
                        _debug_log(e)
                        continue
                    finally:
                        prev_index, prev_score_onset = curr_index, curr_score_onset

            _debug_log("----------------------------------------")

        _debug_log("========================================")
