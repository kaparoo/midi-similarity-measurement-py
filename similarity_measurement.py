# -*- coding: utf-8 -*-

from absl import app
from absl import flags

import csv
import dataset_loader
import numpy as np
import pathlib
import similarity
import tqdm
import warnings

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

_CSV_HEADER = ["Euclidean Similarity", "Timewarping Similarity", "Length ratio"]


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

    with open(save_root / "config.txt", "w", encoding="utf8") as config, open(
        save_root / "pos.csv", "w", encoding="utf8"
    ) as pos_csv, open(save_root / "neg.csv", "w", encoding="utf8") as neg_csv:
        pos_csvfile = csv.writer(pos_csv, delimiter=",", quotechar="|")
        pos_csvfile.writerow(_CSV_HEADER)
        neg_csvfile = csv.writer(neg_csv, delimiter=",", quotechar="|")
        neg_csvfile.writerow(_CSV_HEADER)

        settling_frame = FLAGS.settling_frame
        compensation_frame = FLAGS.compensation_frame
        slice_duration = FLAGS.slice_duration
        expansion_rate = FLAGS.expansion_rate
        frame_per_second = FLAGS.frame_per_second
        queue_size = FLAGS.queue_size
        num_samples = FLAGS.num_samples

        config.writelines(
            [
                f"dataset_root: {str(dataset_root)}\n",
                f"num_samples: {num_samples}\n"
                f"frame_per_second: {frame_per_second}(Hz)\n",
                f"slice_duration: {slice_duration}(sec)\n",
                f"expansion_rate: {expansion_rate}\n",
                f"settling_frame: {settling_frame}\n",
                f"compensation_frame: {compensation_frame}\n",
            ]
        )

        dataset = dataset_loader.spawn(
            dataset_root=dataset_root,
            slice_duration=slice_duration,
            expansion_rate=expansion_rate,
            frame_per_second=frame_per_second,
            shuffle=True,
        )

        pos_similarities = []
        neg_similarities = []

        prev_perfs = [None] * queue_size

        for sample_idx in tqdm.tqdm(range(num_samples)):
            score, perf, (head, tail) = next(dataset)
            score_len = score.shape[-1]
            perf_len = perf.shape[-1]

            (
                pos_euclidean_similarity,
                pos_timewarping_similarity,
                _,
            ) = similarity.score_similarity(
                score, perf, settling_frame, compensation_frame
            )
            pos_length_ratio = perf_len / (score_len + 1e-7)
            pos_similarities.append(
                (
                    pos_euclidean_similarity,
                    pos_timewarping_similarity,
                    pos_length_ratio,
                )
            )
            pos_csvfile.writerow(
                [
                    pos_euclidean_similarity,
                    pos_timewarping_similarity,
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
                    neg_euclidean_similarity,
                    neg_timewarping_similarity,
                    _,
                ) = similarity.score_similarity(
                    score, prev_perf, settling_frame, compensation_frame
                )
                neg_length_ratio = prev_perf_len / (score_len + 1e-7)
                neg_similarities.append(
                    (
                        neg_euclidean_similarity,
                        neg_timewarping_similarity,
                        neg_length_ratio,
                    )
                )
                neg_csvfile.writerow(
                    [
                        neg_euclidean_similarity,
                        neg_timewarping_similarity,
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


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    app.run(main)
