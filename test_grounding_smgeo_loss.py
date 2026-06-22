import unittest

import torch

from grounding.adapters import GroundingOutput
from grounding.losses import compute_grounding_loss


class SMGeoLossTest(unittest.TestCase):
    def test_smgeo_uses_anchor_free_heatmap_and_normalized_bbox_loss(self):
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
                "bbox_weight": 5.0,
                "moe_entropy_weight": 0.01,
            },
        }

        losses = compute_grounding_loss(output, {"bbox": target_bbox}, torch.empty(0), cfg)

        self.assertLess(losses.bbox.item(), 1.0)
        self.assertGreater(losses.heatmap.item(), 0.0)
        expected_total = losses.heatmap + 5.0 * losses.bbox - 0.01 * output.moe_entropy
        self.assertTrue(torch.allclose(losses.total, expected_total))


if __name__ == "__main__":
    unittest.main()
