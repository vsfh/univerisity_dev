import hashlib
from pathlib import Path

import torch

from test_unify import FeatureBundle
from test_unify_ground import (
    GROUND_MODEL_TYPES,
    build_ground_metrics,
    build_parser,
)


EXPECTED_TEST_UNIFY_SHA256 = "544b3fd8ae516485c2eb5821042a96df85daddec2e4037e641e098520b791259"


def test_original_test_unify_is_unchanged():
    digest = hashlib.sha256(Path("test_unify.py").read_bytes()).hexdigest()
    assert digest == EXPECTED_TEST_UNIFY_SHA256


def test_parser_accepts_all_ground_model_types():
    parser = build_parser()
    for model_type in GROUND_MODEL_TYPES:
        args = parser.parse_args(["--model-types", model_type])
        assert args.model_types == [model_type]


def test_ground_metrics_reuses_test_unify_schema():
    bundle = FeatureBundle(
        query_feats=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        query_labels=["10", "20"],
        query_drone_paths=["10/150_0.png", "20/150_0.png"],
        query_satellite_paths=["10.png", "20.png"],
        query_heights=[150, 150],
        query_angles=[0, 0],
        iou_values=[0.5, 0.25],
        center_distances=[2.0, 4.0],
        gallery_labels=["10", "20"],
        gallery_satellite_paths=["10.png", "20.png"],
        gallery_feats=torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
    )

    metrics = build_ground_metrics(
        model_type="det",
        checkpoint_path="/tmp/det/last.pth",
        bundle=bundle,
        include_map={},
        device=torch.device("cpu"),
        candidate_size=100,
        seed=43,
        test_crop_ratio=1.0,
        subset_heights=[150],
        subset_angles=[0],
        include_query_records=False,
    )

    assert metrics["model_type"] == "det"
    assert metrics["overall"]["recall@1"] == 1.0
    assert metrics["overall"]["mean_iou"] == 0.375
    assert metrics["retrieval"]["recall@1"] == 1.0
    assert metrics["query_records"] == []
    assert metrics["per_height"][0]["height"] == 150


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"passed {len(tests)} tests")
