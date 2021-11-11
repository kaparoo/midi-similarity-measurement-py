# -*- coding: utf-8 -*-

from pathlib import Path
from typing import List

__all__ = ["Annotation"]


class Annotation(object):

    __slots__ = ["path", "annotation_path", "onsets"]

    def __init__(self, root_path, midi_name):
        self.path = Path(root_path)
        self.annotation_path = self.path / f"{midi_name}_annotations.txt"
        self.onsets: List[float] = []

        with open(self.annotation_path, "rt") as f:
            lines = f.readlines()
            for line in lines:
                onset = float(line.split("\t")[0])
                self.onsets.append(onset)

    def __iter__(self):
        return AnnotationIterator(self)

    def __getitem__(self, ind) -> float:
        return self.onsets[ind]

    def __len__(self) -> int:
        return len(self.onsets)


class AnnotationIterator:
    def __init__(self, annotation: Annotation):
        self.annotation = annotation
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self) -> float:
        onsets = self.annotation.onsets
        if self.idx >= len(onsets):
            raise StopIteration
        onset = onsets[self.idx]
        self.idx += 1
        return onset
