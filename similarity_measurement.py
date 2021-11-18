# -*- coding: utf-8 -*-

from absl import app
from absl import flags

import csv
import dataset_loader
import metadata
import numpy as np
import pathlib
import similarity
import tqdm
from typing import List, Tuple
import util

FLAGS = flags.FLAGS
flags.DEFINE_string("dataset_root", None, "", required=True)
flags.DEFINE_string("save_root", None, "", required=True)
flags.DEFINE_float("slice_duration", 5.0, "", lower_bound=1.0)
flags.DEFINE_float("expansion_rate", 1.5, "", lower_bound=1.0)
flags.DEFINE_integer("num_samples", 100, "", lower_bound=1)
flags.DEFINE_integer("frame_per_second", 20, "", lower_bound=20)
flags.DEFINE_integer("settling_frame", 10, "", lower_bound=1)
flags.DEFINE_integer("compensation_frame", 0, "", lower_bound=0)
flags.DEFINE_integer("queue_size", 8, "", lower_bound=8)
flags.DEFINE_bool("use_subsequence_dtw", True, "")
flags.DEFINE_bool("use_decay_for_histogram", True, "")

_CSV_HEADER = ["Histogram distance", "Timewarping distance", "Length ratio"]


def main(_):
    dataset_root = pathlib.Path(FLAGS.dataset_root)
    if not (dataset_root.exists() and dataset_root.is_dir()):
        raise FileNotFoundError(dataset_root)

    save_root = pathlib.Path(FLAGS.save_root)
    if not save_root.exists():
        save_root.mkdir(parents=True, exist_ok=True)
    elif not save_root.is_dir():
        raise FileExistsError(save_root)

    npz_root = save_root / "npz"
    if not npz_root.exists():
        npz_root.mkdir(parents=True, exist_ok=True)
    elif not npz_root.is_dir():
        raise FileExistsError(npz_root)

    with open(save_root / "pos.csv", "w", encoding="utf8") as pos_csv, open(
        save_root / "neg.csv", "w", encoding="utf8"
    ) as neg_csv:
        pos_csvfile = csv.writer(pos_csv, delimiter=",", quotechar="|")
        pos_csvfile.writerow(_CSV_HEADER)
        neg_csvfile = csv.writer(neg_csv, delimiter=",", quotechar="|")
        neg_csvfile.writerow(_CSV_HEADER)

        num_samples = FLAGS.num_samples
        queue_size = FLAGS.queue_size
        config = metadata.Metadata(
            dataset_root=str(dataset_root),
            frame_per_second=FLAGS.frame_per_second,
            slice_duration=FLAGS.slice_duration,
            expansion_rate=FLAGS.expansion_rate,
            settling_frame=FLAGS.settling_frame,
            compensation_frame=FLAGS.compensation_frame,
            use_subsequence_dtw=FLAGS.use_subsequence_dtw,
            use_decay_for_histogram=FLAGS.use_decay_for_histogram,
        )
        config.save(filepath=save_root / "config.json")

        dataset = dataset_loader.spawn(
            dataset_root=dataset_root,
            slice_duration=config.slice_duration,
            expansion_rate=config.expansion_rate,
            frame_per_second=config.frame_per_second,
            shuffle=True,
        )

        pos_similarities: List[Tuple[float, float, float]] = []
        neg_similarities: List[Tuple[float, float, float]] = []

        prev_perfs: List[np.ndarray] = [None] * queue_size

        try:
            for sample_idx in tqdm.trange(
                num_samples, desc="Measuring similarities..."
            ):
                score, perf, (head, tail) = next(dataset)
                score_len = score.shape[-1]
                perf_len = perf.shape[-1]

                (
                    pos_histogram_distance,
                    pos_timewarping_distance,
                    _,
                ) = similarity.measure(
                    score,
                    perf,
                    config.settling_frame,
                    config.compensation_frame,
                    config.use_subsequence_dtw,
                    config.use_decay_for_histogram,
                )
                pos_length_ratio = perf_len / (score_len + 1e-7)
                pos_similarities.append(
                    (
                        pos_histogram_distance,
                        pos_timewarping_distance,
                        pos_length_ratio,
                    )
                )
                pos_csvfile.writerow(
                    [
                        pos_histogram_distance,
                        pos_timewarping_distance,
                        pos_length_ratio,
                    ]
                )

                np.savez(
                    npz_root / f"pos_{sample_idx}.npz",
                    score=score,
                    perf=perf,
                    alignment=(head, tail),
                )

                if isinstance(prev_perfs[0], np.ndarray):
                    prev_perf = prev_perfs[0]
                    prev_perf_len = prev_perf.shape[-1]
                    (
                        neg_histogram_distance,
                        neg_timewarping_distance,
                        _,
                    ) = similarity.measure(
                        score,
                        prev_perf,
                        config.settling_frame,
                        config.compensation_frame,
                        config.use_subsequence_dtw,
                        config.use_decay_for_histogram,
                    )
                    neg_length_ratio = prev_perf_len / (score_len + 1e-7)
                    neg_similarities.append(
                        (
                            neg_histogram_distance,
                            neg_timewarping_distance,
                            neg_length_ratio,
                        )
                    )
                    neg_csvfile.writerow(
                        [
                            neg_histogram_distance,
                            neg_timewarping_distance,
                            neg_length_ratio,
                        ]
                    )

                    np.savez(
                        npz_root / f"neg_{sample_idx-queue_size}.npz",
                        score=score,
                        perf=prev_perf,
                    )

                prev_perfs.pop(0)
                prev_perfs.append(perf)

        except StopIteration:
            print(f"Loading dataset is finished at iteration {sample_idx}.")

        pos_similarities = np.array(pos_similarities)
        neg_similarities = np.array(neg_similarities)

        util.save_scatter_plots(save_root, pos_similarities, neg_similarities)


if __name__ == "__main__":
    app.run(main)
