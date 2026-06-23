import unittest
from unittest.mock import patch

import torch

from grounding.adapters import GroundingOutput
from grounding.losses import compute_grounding_loss
from grounding.registry import _build_smgeo


class SMGeoLossTest(unittest.TestCase):
    def test_smgeo_uses_official_anchor_free_masked_bbox_loss(self):
        heatmap_logits = torch.zeros(1, 1, 2, 2)
        bbox_raw = torch.zeros(1, 4, 2, 2)
        target_bbox = torch.tensor([[160.0, 160.0, 480.0, 480.0]], dtype=torch.float32)
        output = GroundingOutput(
            device=torch.device("cpu"),
            image_wh=(640, 640),
            pred_bbox=torch.tensor([[0.0, 0.0, 640.0, 640.0]], dtype=torch.float32),
            heatmap=heatmap_logits,
            bbox_raw=bbox_raw,
            moe_entropy=torch.tensor(0.25),
        )
        cfg = {
            "model": {"type": "smgeo"},
            "loss": {
                "heatmap_weight": 0.7,
                "bbox_weight": 0.3,
                "moe_entropy_weight": 0.01,
            },
        }

        losses = compute_grounding_loss(output, {"bbox": target_bbox}, torch.empty(0), cfg)

        self.assertAlmostEqual(losses.bbox.item(), 1.95, places=4)
        self.assertGreater(losses.heatmap.item(), 0.0)
        expected_total = 0.7 * losses.heatmap + 0.3 * losses.bbox - 0.01 * output.moe_entropy
        self.assertTrue(torch.allclose(losses.total, expected_total))

    def test_smgeo_builder_loads_configured_checkpoint(self):
        checkpoint_path = "/tmp/fake-smgeo.pth"
        with patch("grounding.legacy.train_sm.load_smgeo_pretrained", return_value={"loaded": 1}) as load_mock:
            _build_smgeo({"model": {"checkpoint": checkpoint_path}})

        self.assertEqual(load_mock.call_count, 1)
        self.assertEqual(load_mock.call_args.args[1], checkpoint_path)


if __name__ == "__main__":
    unittest.main()
