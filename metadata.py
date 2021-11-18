from dataclass_base import DataclassBase


class Metadata(DataclassBase):
    JSON_KEY = "Metadata"

    __slots__ = [
        "dataset_root",
        "frame_per_second",
        "slice_duration",
        "expansion_rate",
        "settling_frame",
        "compensation_frame",
        "use_subsequence_dtw",
        "use_decay_for_histogram",
    ]

    def __init__(
        self,
        dataset_root: str,
        frame_per_second: int,
        slice_duration: int,
        expansion_rate: float,
        settling_frame: int,
        compensation_frame: int,
        use_subsequence_dtw: bool,
        use_decay_for_histogram: bool,
    ) -> None:
        self.dataset_root = dataset_root
        self.frame_per_second = frame_per_second
        self.slice_duration = slice_duration
        self.expansion_rate = expansion_rate
        self.settling_frame = settling_frame
        self.compensation_frame = compensation_frame
        self.use_subsequence_dtw = use_subsequence_dtw
        self.use_decay_for_histogram = use_decay_for_histogram
