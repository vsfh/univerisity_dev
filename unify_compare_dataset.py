"""Dataset adapters for fair UnifyGeo/TransGeo/SigLIP comparisons.

The shared :mod:`dataset` implementation remains the source of annotations,
I/O, preprocessing, and returned fields. The subclasses only lock explicitly
named comparison protocols; they do not modify the shared dataset class.
"""

from typing import Tuple

from dataset import ShiftedSatelliteDroneDataset


class UnifiedSiglipSuppComparisonDataset(ShiftedSatelliteDroneDataset):
    """Lock the shared dataset to ``unified_siglip_supp.py`` data semantics.

    The comparison models may use model-specific image normalization, but they
    must see the same sample identities, train boxes and satellite crop policy
    as Unified SigLIP Supp.  Keeping this contract in a subclass avoids any
    change to :mod:`dataset` and prevents comparison-only defaults from leaking
    into the shared implementation.
    """

    def __init__(self, *args, split: str, **kwargs):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'.")

        # unified_siglip_supp.py passes neither option, so these are the exact
        # shared-dataset defaults used by that baseline.
        requested_crop_range = kwargs.pop("train_crop_ratio_range", None)
        requested_bbox_scale = float(kwargs.pop("train_bbox_scale", 1.0))
        if requested_crop_range is not None:
            raise ValueError(
                "UnifiedSiglipSuppComparisonDataset fixes train_crop_ratio_range=None. "
                "Use JointGeoTrainingDataset for comparison-specific crop jitter."
            )
        if requested_bbox_scale != 1.0:
            raise ValueError(
                "UnifiedSiglipSuppComparisonDataset fixes train_bbox_scale=1.0. "
                "Use JointGeoTrainingDataset for rescaled synthetic boxes."
            )

        super().__init__(
            *args,
            split=split,
            train_crop_ratio_range=None,
            train_bbox_scale=1.0,
            **kwargs,
        )


class JointGeoTrainingDataset(ShiftedSatelliteDroneDataset):
    """Train-split adapter without changing the shared dataset class."""

    def __init__(
        self,
        *args,
        train_crop_ratio_range: Tuple[float, float] = (0.6, 1.0),
        train_bbox_scale: float = 2.0,
        **kwargs,
    ):
        requested_split = kwargs.pop("split", "train")
        if requested_split != "train":
            raise ValueError(
                "JointGeoTrainingDataset is training-only; use "
                "ShiftedSatelliteDroneDataset for validation and testing."
            )
        super().__init__(
            *args,
            split="train",
            train_crop_ratio_range=train_crop_ratio_range,
            train_bbox_scale=train_bbox_scale,
            **kwargs,
        )
