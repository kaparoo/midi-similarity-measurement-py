# -*- coding: utf-8 -*-


from typing import Dict, List, Tuple, Union


__all__ = ["score_alignment"]


def score_alignment(
    ground_truth: Tuple[int, int], prediction: Tuple[int, int], length: int
) -> Dict[str, Union[List[List[float]], float]]:
    true_head, true_tail = ground_truth
    pred_head, pred_tail = prediction

    min_head = min(true_head, pred_head)
    max_head = max(true_head, pred_head)
    min_tail = min(true_tail, pred_tail)
    max_tail = max(true_tail, pred_tail)

    if min_tail <= max_head:
        return {
            "confusion_matrix": [[0.0, 0.0], [0.0, 0.0]],
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "iou": 0.0,
        }

    true_positive = (min_tail - max_head + 1) / length
    true_negative = (length - max_tail + min_head) / length

    head_diff = (max_head - min_head) / length
    tail_diff = (max_tail - min_tail - 1) / length if max_tail != min_tail else 0.0

    false_positive = 0.0
    false_negative = 0.0
    if pred_head < true_head:
        false_positive += head_diff
    elif true_head < pred_head:
        false_negative += head_diff
    if true_tail < pred_tail:
        false_positive += tail_diff
    elif pred_tail < true_tail:
        false_negative += tail_diff

    confusion_matrix = [
        [true_positive, false_positive],
        [false_negative, true_negative],
    ]

    accuracy = true_positive + true_negative / (
        true_positive + true_negative + false_positive + false_negative
    )
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score = 0.0
    if precision + recall != 0:
        f1_score = 2 * (precision * recall) / (precision + recall)

    iou = true_positive / (true_positive + false_positive + false_negative)

    return {
        "confusion_matrix": confusion_matrix,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "iou": iou,
    }
