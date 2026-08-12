import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from dataset import ShiftedSatelliteDroneDataset
from test_unify import (
    FeatureBundle,
    load_checkpoint,
    resolve_trans_drone_size,
    score_retrieval_and_uiou,
    score_unify_query,
    summarize_records,
)
from train_trans import (
    DRONE_SIZE as TRANS_DRONE_SIZE,
    SAT_SIZE as TRANS_SAT_SIZE,
    average_accumulated_gradients,
    asam_perturb,
    asam_restore,
    exhaustive_soft_margin_triplet,
    losses_are_finite,
)
from train_uni import (
    IdentityBalancedBatchSampler,
    LocalizationDecoder,
    RetrievalMemoryQueue,
    bbox_regression_loss,
    multi_positive_cross_entropy,
)
from unify_compare_dataset import (
    JointGeoTrainingDataset,
    UnifiedSiglipSuppComparisonDataset,
)


class _FakeDataset:
    def __init__(self, identity_count: int, samples_per_identity: int):
        self.samples = [
            {"satellite_id": identity}
            for identity in range(identity_count)
            for _ in range(samples_per_identity)
        ]


class TrainUnifyCompareTests(unittest.TestCase):
    @staticmethod
    def _metric_record(
        *,
        top1_correct: bool,
        center_distance: float,
        iou: float = 0.5,
    ):
        return {
            "top1_correct": top1_correct,
            "top5_correct": top1_correct,
            "top10_correct": top1_correct,
            "iou": iou,
            "uIoU": iou if top1_correct else 0.0,
            "center_distance": center_distance,
        }

    def test_ucde_excludes_retrieval_failures_from_sum_and_denominator(self):
        records = [
            self._metric_record(top1_correct=True, center_distance=10.0),
            self._metric_record(top1_correct=False, center_distance=1000.0),
            self._metric_record(top1_correct=True, center_distance=30.0),
        ]
        summary = summarize_records(records)
        self.assertAlmostEqual(summary["mean_center_distance"], 1040.0 / 3.0, places=4)
        self.assertAlmostEqual(summary["uCDE"], 20.0, places=6)
        self.assertEqual(summary["uCDE_num_samples"], 2)

    def test_ucde_is_none_when_a_group_has_no_retrieval_success(self):
        summary = summarize_records(
            [self._metric_record(top1_correct=False, center_distance=123.0)]
        )
        self.assertIsNone(summary["uCDE"])
        self.assertEqual(summary["uCDE_num_samples"], 0)

    def test_ucde_shared_scoring_path_covers_requested_model_types(self):
        query_feats = torch.eye(2)
        common_kwargs = {
            "query_feats": query_feats,
            "query_labels": ["0", "1"],
            "query_drone_paths": ["0/250_0.png", "1/250_0.png"],
            "query_satellite_paths": ["0.png", "1.png"],
            "query_heights": [250, 250],
            "query_angles": [0, 0],
            "iou_values": [0.4, 0.6],
            "center_distances": [10.0, 20.0],
            "gallery_labels": ["0", "1"],
            "gallery_satellite_paths": ["0.png", "1.png"],
        }
        for model_type in ("encoder_test", "unify_geo", "trans_geo"):
            if model_type == "encoder_test":
                bundle = FeatureBundle(
                    **common_kwargs,
                    gallery_feats=torch.eye(2).unsqueeze(1),
                )
            else:
                bundle = FeatureBundle(
                    **common_kwargs,
                    gallery_feats=torch.eye(2),
                    query_detail_feats=torch.eye(2),
                    gallery_detail_feats=torch.eye(2).view(2, 2, 1, 1),
                    unify_logit_scale=1.0,
                    unify_temperature=1.0,
                )
            scored = score_retrieval_and_uiou(
                model_type=model_type,
                bundle=bundle,
                include_map={},
                device=torch.device("cpu"),
                candidate_size=None,
                unify_score_mode="rerank",
                sampling_seed=43,
            )
            summary = summarize_records(scored["query_records"])
            self.assertEqual(summary["top1_hits"], 2, model_type)
            self.assertAlmostEqual(summary["uCDE"], 15.0, places=6, msg=model_type)
            self.assertEqual(summary["uCDE_num_samples"], 2, model_type)

    def test_global_unify_scoring_does_not_require_detail_cache(self):
        bundle = FeatureBundle(
            query_feats=torch.eye(2),
            query_labels=["0", "1"],
            query_drone_paths=["0.png", "1.png"],
            query_satellite_paths=["0.png", "1.png"],
            query_heights=[250, 250],
            query_angles=[0, 0],
            iou_values=[0.4, 0.6],
            center_distances=[10.0, 20.0],
            gallery_labels=["0", "1"],
            gallery_satellite_paths=["0.png", "1.png"],
            gallery_feats=torch.eye(2),
            query_detail_feats=None,
            gallery_detail_feats=None,
            unify_logit_scale=1.0,
        )
        scores = score_unify_query(
            bundle=bundle,
            query_index=0,
            candidate_indices=[0, 1],
            device=torch.device("cpu"),
            score_mode="global",
        )
        self.assertTrue(torch.equal(scores, torch.tensor([1.0, 0.0])))

    def test_transgeo_fair_input_geometry_matches_unified_siglip_supp(self):
        self.assertEqual(TRANS_DRONE_SIZE, (224, 224))
        self.assertEqual(TRANS_SAT_SIZE, (768, 432))

    def test_training_dataset_is_an_adapter_of_the_shared_dataset(self):
        self.assertTrue(issubclass(JointGeoTrainingDataset, ShiftedSatelliteDroneDataset))

    def test_unified_siglip_comparison_dataset_locks_shared_defaults(self):
        with mock.patch.object(
            ShiftedSatelliteDroneDataset,
            "__init__",
            return_value=None,
        ) as parent_init:
            UnifiedSiglipSuppComparisonDataset(
                processor=object(),
                processor_sat=object(),
                tokenizer=object(),
                split="train",
            )

        forwarded = parent_init.call_args.kwargs
        self.assertEqual(forwarded["split"], "train")
        self.assertIsNone(forwarded["train_crop_ratio_range"])
        self.assertEqual(forwarded["train_bbox_scale"], 1.0)

    def test_unified_siglip_comparison_dataset_rejects_comparison_overrides(self):
        with self.assertRaises(ValueError):
            UnifiedSiglipSuppComparisonDataset(
                processor=object(),
                processor_sat=object(),
                tokenizer=object(),
                split="train",
                train_bbox_scale=2.0,
            )

    def test_identity_sampler_keeps_unique_ids_and_does_not_drop_each_round(self):
        dataset = _FakeDataset(identity_count=20, samples_per_identity=4)
        sampler = IdentityBalancedBatchSampler(dataset, batch_size=16, drop_last=True, seed=7)
        batches = list(sampler)
        flattened = [index for batch in batches for index in batch]

        self.assertEqual(len(batches), 5)
        self.assertEqual(len(flattened), len(dataset.samples))
        self.assertEqual(len(flattened), len(set(flattened)))
        for batch in batches:
            identities = [dataset.samples[index]["satellite_id"] for index in batch]
            self.assertEqual(len(identities), len(set(identities)))

    def test_multi_positive_loss_accepts_duplicate_satellite_ids(self):
        logits = torch.tensor(
            [[8.0, 7.0, -4.0], [7.0, 8.0, -4.0], [-3.0, -3.0, 8.0]],
            requires_grad=True,
        )
        identities = torch.tensor([10, 10, 11])
        loss = multi_positive_cross_entropy(logits, identities, identities)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)

    def test_memory_queue_rejects_nonfinite_feature_rows(self):
        queue = RetrievalMemoryQueue(capacity=8)
        query = torch.tensor([[1.0, 0.0], [float("nan"), 1.0]])
        aerial = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        queue.enqueue(query, aerial, torch.tensor([10, 11]))
        queued_query, queued_aerial, queued_ids = queue.get(torch.device("cpu"))
        self.assertEqual(len(queue), 1)
        self.assertTrue(torch.isfinite(queued_query).all())
        self.assertTrue(torch.isfinite(queued_aerial).all())
        self.assertEqual(queued_ids.tolist(), [10])

    def test_nonfinite_loss_is_detected_before_backward(self):
        self.assertTrue(losses_are_finite(torch.tensor(1.0), {"retrieval": torch.tensor(2.0)}))
        self.assertFalse(
            losses_are_finite(torch.tensor(float("nan")), {"retrieval": torch.tensor(2.0)})
        )

    def test_transgeo_exhaustive_triplet_prefers_aligned_pairs(self):
        query = torch.eye(4, requires_grad=True)
        aligned = torch.eye(4)
        misaligned = torch.roll(aligned, shifts=1, dims=0)
        identities = torch.arange(4)
        aligned_loss = exhaustive_soft_margin_triplet(query, aligned, identities, alpha=10.0)
        misaligned_loss = exhaustive_soft_margin_triplet(query, misaligned, identities, alpha=10.0)
        self.assertLess(float(aligned_loss), float(misaligned_loss))
        aligned_loss.backward()
        self.assertIsNotNone(query.grad)

    def test_bbox_loss_backpropagates_to_heatmap_and_regression(self):
        heatmap = torch.randn(2, 1, 6, 8, requires_grad=True)
        bbox_raw = torch.randn(2, 4, 6, 8, requires_grad=True)
        target = torch.tensor([[50.0, 40.0, 130.0, 120.0], [210.0, 80.0, 330.0, 200.0]])
        loss, _, _ = bbox_regression_loss(
            heatmap,
            bbox_raw,
            target,
            image_wh=(384, 216),
            iou_weight=2.0,
        )
        loss.backward()
        self.assertGreater(float(heatmap.grad.abs().sum()), 0.0)
        self.assertGreater(float(bbox_raw.grad.abs().sum()), 0.0)

    def test_bbox_head_starts_near_true_box_scale(self):
        decoder = LocalizationDecoder(detail_dim=8)
        initial_size = decoder.bbox_head[-1].bias[2:].sigmoid()
        self.assertTrue(torch.allclose(initial_size, torch.full_like(initial_size, 0.15), atol=1e-4))

    def test_asam_perturbation_is_reversible(self):
        model = nn.Linear(3, 2)
        before = model.weight.detach().clone()
        model(torch.ones(2, 3)).square().mean().backward()
        perturbations = asam_perturb(model, rho=0.05)
        self.assertFalse(torch.equal(before, model.weight.detach()))
        asam_restore(perturbations)
        self.assertTrue(torch.allclose(before, model.weight.detach()))

    def test_accumulated_gradients_are_averaged(self):
        model = nn.Linear(2, 1, bias=False)
        model.weight.grad = torch.tensor([[6.0, -2.0]])
        average_accumulated_gradients(model, microbatch_count=2)
        self.assertTrue(
            torch.equal(model.weight.grad, torch.tensor([[3.0, -1.0]]))
        )

    def test_training_checkpoint_payload_loads_in_test_unify(self):
        source = nn.Linear(3, 2)
        target = nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = Path(tmp_dir) / "checkpoint.pth"
            torch.save(
                {"model": source.state_dict(), "epoch": 1, "metric": np.float64(0.5)},
                checkpoint,
            )
            load_checkpoint(target, str(checkpoint))
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            self.assertTrue(torch.equal(source_param, target_param))

    def test_transgeo_test_geometry_comes_from_saved_data_contract(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = Path(tmp_dir) / "last.pth"
            config = Path(tmp_dir) / "train_config.json"
            config.write_text(
                '{"data_contract": {"drone_size_wh": [224, 224]}}',
                encoding="utf-8",
            )
            self.assertEqual(resolve_trans_drone_size(str(checkpoint)), (224, 224))

    def test_transgeo_old_geometry_is_inferred_from_position_tokens(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = Path(tmp_dir) / "last.pth"
            torch.save(
                {"model": {"query_encoder.backbone.pos_embed": torch.zeros(1, 258, 384)}},
                checkpoint,
            )
            self.assertEqual(resolve_trans_drone_size(str(checkpoint)), (256, 256))


if __name__ == "__main__":
    unittest.main()
