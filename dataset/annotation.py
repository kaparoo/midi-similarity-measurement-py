# -*- coding: utf-8 -*-

from os import PathLike
from pathlib import Path
from typing import List

__all__ = ["Annotation"]


class Annotation(object):

    __slots__ = ["_path", "_onsets", "_length"]

    def __init__(self, path: PathLike, prefix: str) -> None:
        self._path = Path(path) / f"{prefix}_annotations.txt"
        self._onsets = []
        with self._path.open(mode="rt", encoding="utf-8") as f:
            for line in f.readlines():
                onset = float(line.split("\t")[0])
                self._onsets.append(onset)
        self._length = len(self._onsets)

    @property
    def path(self) -> str:
        return str(self._path)

    @property
    def onsets(self) -> List[float]:
        return self._onsets

    @property
    def length(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> float:
        return self._onsets[idx]

    def __iter__(self):
        return AnnotationIter(self)

    def __len__(self):
        return self._length


class AnnotationIter(object):

    __slots__ = ["_cursor", "_onsets", "_length"]

    def __init__(self, annotation: Annotation) -> None:
        self._cursor = 0
        self._onsets = annotation.onsets
        self._length = annotation.length

    def __iter__(self):
        return self

    def __next__(self) -> float:
        if self._cursor >= self._length:
            raise StopIteration
        onset = self._onsets[self._cursor]
        self._cursor += 1
        return onset


if __name__ == "__main__":
    path = "../dataset/newbie-dataset/Clementi/sonatina_op36_no3_pt1"
    annotation = Annotation(path, prefix="score")
    onsets = annotation.onsets
    print(f"onsets: {onsets} ({len(onsets)})")  # [0.0, 0.55, ..., 53.3] (104)
