# -*- coding: utf-8 -*-

import dataset
import midi_unit
import numpy as np
from os import PathLike
from similarity import measure, Similarity
from tqdm import tqdm
from typing import List, Optional, Tuple


__all__ = ["run"]


def run(
    dataset_root: PathLike,
    score_prefix: str = "score",
    num_samples: Optional[int] = None,
    queue_size: int = 8,
    settling_frame: int = 2,
    frames_per_second: int = 20,
    slice_duration: Tuple[float, Tuple[float, float]] = 1.0,
    expansion_rate: Tuple[float, Tuple[float, float]] = 1.0,
    subsequence: bool = True,
    shuffle: bool = True,
    verbose: bool = False,
) -> Tuple[List[Similarity], List[Similarity]]:
    generator = dataset.new_generator(
        dataset_root,
        score_prefix,
        slice_duration,
        expansion_rate,
        frames_per_second,
        shuffle=shuffle,
        verbose=verbose,
    )

    if verbose:
        generator = tqdm(generator, total=num_samples)

    decay_fn = midi_unit.make_decay_fn("linear", settling_frame)

    pos_similarities: List[Similarity] = []
    neg_similarities: List[Similarity] = []
    prev_perfs: List[np.ndarray] = [None] * queue_size

    try:
        if not verbose:
            for idx, (score, perf, _) in enumerate(generator):
                if idx == num_samples:
                    raise StopIteration

                pos_similarity = measure(
                    score, perf, decay_fn=decay_fn, subsequence=subsequence
                )
                pos_similarities.append(pos_similarity)
                if isinstance(prev_perf := prev_perfs[0], np.ndarray):
                    neg_similarity = measure(
                        score, prev_perf, decay_fn=decay_fn, subsequence=subsequence
                    )
                    neg_similarities.append(neg_similarity)
                prev_perfs.pop(0)
                prev_perfs.append(perf)
        else:
            description = "[%d] Histogram: %.4f, Timewarping: %.4f, Length ratio: %.4f"
            generator.set_description(description % (0, -1, -1, -1))
            for idx, (
                (score, perf, _),
                (perf_path, slice_duration, expansion_rate),
            ) in enumerate(generator):
                if idx == num_samples:
                    raise StopIteration

                pos_similarity = measure(
                    score, perf, decay_fn=decay_fn, subsequence=subsequence
                )
                pos_similarities.append(pos_similarity)
                if isinstance(prev_perf := prev_perfs[0], np.ndarray):
                    neg_similarity = measure(
                        score, prev_perf, decay_fn=decay_fn, subsequence=subsequence
                    )
                    neg_similarities.append(neg_similarity)
                prev_perfs.pop(0)
                prev_perfs.append(perf)
                if verbose:
                    generator.set_description(description % (idx, *pos_similarity))
                    generator.set_postfix_str(
                        f"{perf_path}, ({slice_duration=:.2f}, {expansion_rate=:.2f})"
                    )

    except StopIteration:
        if verbose:
            print(f"Loading dataset is finished at iteration {idx}.")
            generator.close()

    return pos_similarities, neg_similarities
