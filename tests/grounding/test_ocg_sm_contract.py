import torch
import torch.nn as nn

from grounding.adapters import SMGeoAdapter
from grounding.legacy.train_ocg import OCGNetLite
from grounding.legacy.train_sm import SMGeoLite


class QueryEncoderStub(nn.Module):
    def forward(self, images, click_maps):
        base = nn.functional.adaptive_avg_pool2d(images.mean(dim=1, keepdim=True), (2, 2))
        click = nn.functional.adaptive_avg_pool2d(click_maps, (2, 2))
        return torch.cat([base, click, base + click, base - click], dim=1)


class ReferenceEncoderStub(nn.Module):
    def forward(self, images):
        base = nn.functional.adaptive_avg_pool2d(images.mean(dim=1, keepdim=True), (2, 2))
        return base.repeat(1, 4, 1, 1)


class CrossAttentionStub(nn.Module):
    def forward(self, query, reference):
        return query + reference.mean(dim=(2, 3), keepdim=True)


class FusionStub(nn.Module):
    def forward(self, query, reference):
        return (query + reference)[:, :2]


class SMBackboneStub(nn.Module):
    out_dim = 4

    def forward(self, query_imgs, sat_imgs):
        query_map = query_imgs[:, 0]
        query_vec = torch.stack(
            [
                query_map.mean(dim=(1, 2)),
                query_map[:, :, : query_map.shape[2] // 2].mean(dim=(1, 2)),
                query_map[:, :, query_map.shape[2] // 2 :].mean(dim=(1, 2)),
                query_map[:, : query_map.shape[1] // 2].mean(dim=(1, 2)),
            ],
            dim=1,
        )
        sat = nn.functional.adaptive_avg_pool2d(sat_imgs.mean(dim=1, keepdim=True), (2, 2))
        sat_feat = sat.repeat(1, 4, 1, 1)
        return query_vec, sat_feat, query_vec.new_tensor(0.5)


class SMHeadStub(nn.Module):
    def forward(self, features):
        return features[:, :1], features


def assert_contract(output):
    assert output["query_embedding"].shape == (2, 4)
    assert output["search_local_features"].shape == (2, 4, 2, 2)
    assert output["matcher_logits"].shape == (2, 8)


def test_ocg_uses_independent_encoders_and_mandatory_matcher():
    query_encoder = QueryEncoderStub()
    reference_encoder = ReferenceEncoderStub()
    model = OCGNetLite(
        channels=4,
        num_heads=2,
        query_encoder=query_encoder,
        reference_encoder=reference_encoder,
        cross_attention=CrossAttentionStub(),
        fusion=FusionStub(),
        projection_dim=4,
    )
    assert model.query_encoder is not model.reference_encoder

    output = model(
        torch.randn(2, 3, 8, 8),
        torch.randn(2, 3, 8, 8),
        query_click=torch.tensor([[0.5, 0.5], [0.25, 0.75]]),
    )
    assert output["pred_anchor"].shape == (2, 45, 2, 2)
    assert_contract(output)


def test_smgeo_uses_click_and_returns_query_guard_contract():
    model = SMGeoLite(
        backbone=SMBackboneStub(),
        head=SMHeadStub(),
        projection_dim=4,
    )
    with torch.no_grad():
        model.query_click_adapter.weight.zero_()
        model.query_click_adapter.weight[0, 3, 0, 0] = 1.0

    query = torch.zeros(2, 3, 8, 8)
    satellite = torch.ones(2, 3, 8, 8)
    center = model(
        query,
        satellite,
        query_click=torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
    )
    corner = model(
        query,
        satellite,
        query_click=torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
    )

    assert_contract(center)
    assert center["heatmap_logits"].shape == (2, 1, 2, 2)
    assert not torch.allclose(center["query_embedding"], corner["query_embedding"])
    torch.testing.assert_close(
        center["gated_search"],
        center["search_native"] * center["spatial_gate"],
    )


def test_smgeo_adapter_decodes_mapping_output():
    model = SMGeoLite(
        backbone=SMBackboneStub(),
        head=SMHeadStub(),
        projection_dim=4,
    )
    adapter = SMGeoAdapter(model, {"model": {"use_angle": False}})
    batch = {
        "target_pixel_values": torch.zeros(2, 3, 8, 8),
        "search_pixel_values": torch.ones(2, 3, 8, 8),
        "query_click": torch.tensor([[0.5, 0.5], [0.25, 0.75]]),
    }
    output = adapter.forward(batch, torch.device("cpu"))

    assert output.pred_bbox.shape == (2, 4)
    assert output.query_embedding.shape == (2, 4)
    assert output.matcher_logits.shape == (2, 8)


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"passed {len(tests)} tests")
