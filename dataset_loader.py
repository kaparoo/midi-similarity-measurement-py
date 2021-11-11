# -*- coding: utf-8 -*-

import numbers

import os
import pathlib

from typing import Any, Final, List, Tuple, Union

__all__ = ["spawn"]

_SCORE_MIDI_FILE: Final[str] = u"midi_score.mid"


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
            raise ValueError()
        else:
            dataset_root = pathlib.Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError()
    elif not dataset_root.is_dir():
        raise FileExistsError()

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
):
    dataset_root, slice_duration, expansion_rate, frame_per_second = _verify_arguments(
        dataset_root, slice_duration, expansion_rate, frame_per_second
    )

    dataset_infos: List[Tuple[pathlib.Path, List[str]]] = []

    for path, _, files in os.walk(dataset_root):
        if _SCORE_MIDI_FILE in files:
            perf_files = [
                file
                for file in files
                if file != _SCORE_MIDI_FILE and os.path.splitext()[-1] != "mid"
            ]

    for path, perf_midi_files in dataset_infos:
        score_midi_path = path / _SCORE_MIDI_FILE

        for perf_midi_file in perf_midi_files:
            perf_midi_path = path / perf_midi_file

            print()
