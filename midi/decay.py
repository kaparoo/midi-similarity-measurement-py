# -*- coding: utf-8 -*-

import numpy as np
from typing import Callable, Optional

__all__ = ["DecayFn", "get_decay_fn"]


DecayFn = Callable[[np.ndarray], np.ndarray]


def linear_decay_factory(settling_frame: int = 10) -> DecayFn:
    def linear_decay(array: np.ndarray) -> np.ndarray:
        decayed_array = np.zeros_like(array, dtype=np.uint8)
        prev_pressed = False
        for idx, marker in enumerate(array):
            curr_velocity = 0
            if marker <= 0:
                prev_pressed = False
            elif marker > 1:
                curr_velocity = settling_frame - 1
                prev_pressed = True
            else:
                if idx == 0:
                    curr_velocity = settling_frame - 1
                elif (prev_velocity := decayed_array[idx - 1]) > 0:
                    curr_velocity = prev_velocity - 1
                elif not prev_pressed:
                    curr_velocity = settling_frame - 1
                prev_pressed = True
            decayed_array[idx] = curr_velocity
        return decayed_array

    return linear_decay


def get_decay_fn(type: Optional[str] = None, settling_frame: int = 10) -> DecayFn:
    if type == "linear":
        return linear_decay_factory(settling_frame)
    else:
        return lambda x: x
