import torch
import torch.nn as nn

from grounding.legacy.ground_cvos import DetGeoLite, LPNGeoLite, SampleGeoLite, TROGeoLite


class MapBackbone(nn.Module):
    def __init__(self, channels=4):
        super().__init__()
        self.conv = nn.Conv2d(3, channels, kernel_size=1)

    def forward(self, images):
        return nn.functional.adaptive_avg_pool2d(self.conv(images), (2, 2))


class DarknetLikeBackbone(MapBackbone):
    def forward(self, images):
        features = super().forward(images)
        return features, features, features


class ConvNeXtLikeBackbone(MapBackbone):
    def forward_features(self, images):
        return super().forward(images)

    def forward_head(self, features, pre_logits=False):
        return features.mean(dim=(2, 3))


class FeaturesOnlyBackbone(MapBackbone):
    def forward(self, images):
        return [super().forward(images)]


class IdentityCrossAttention(nn.Module):
    def forward(self, x, context):
        assert context.ndim == 3
        return x


def assert_query_guard_contract(output, batch_size=2):
    assert isinstance(output, dict)
    assert output["pred_anchor"].shape[0] == batch_size
    assert output["query_embedding"].shape == (batch_size, 4)
    assert output["search_local_features"].shape == (batch_size, 4, 2, 2)
    assert output["matcher_logits"].shape == (batch_size, batch_size * 4)
    assert torch.isfinite(output["matcher_logits"]).all()


def test_detgeo_uses_separate_towers_and_returns_query_guard_contract():
    query_backbone = MapBackbone()
    reference_backbone = DarknetLikeBackbone()
    model = DetGeoLite(
        emb_size=4,
        query_backbone=query_backbone,
        reference_backbone=reference_backbone,
        query_feature_dim=4,
        reference_feature_dim=4,
        projection_dim=4,
        load_reference_weights=False,
    )
    assert model.query_resnet is not model.reference_darknet

    output = model(
        torch.randn(2, 3, 8, 8),
        torch.randn(2, 3, 8, 8),
        query_click=torch.tensor([[0.5, 0.5], [0.25, 0.75]]),
    )
    assert_query_guard_contract(output)


def test_trogeo_returns_query_guard_contract_with_explicit_click():
    shared = MapBackbone()
    model = TROGeoLite(
        emb_size=4,
        backbone=shared,
        cross_attention=IdentityCrossAttention(),
        projection_dim=4,
    )
    output = model(
        torch.randn(2, 3, 8, 8),
        torch.randn(2, 3, 8, 8),
        query_click=torch.tensor([[0.5, 0.5], [0.25, 0.75]]),
    )
    assert_query_guard_contract(output)


def test_sample4geo_keeps_shared_backbone_and_adds_symmetric_retrieval_loss():
    shared = ConvNeXtLikeBackbone()
    model = SampleGeoLite(
        emb_size=4,
        backbone=shared,
        cross_attention=IdentityCrossAttention(),
        projection_dim=4,
    )
    assert model.query_model is model.reference_model
    output = model(torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8))

    assert_query_guard_contract(output)
    assert output["paper_aux_losses"]["sample4geo_symmetric_infonce"].ndim == 0


def test_lpn_uses_ring_descriptor_and_returns_query_guard_contract():
    shared = FeaturesOnlyBackbone()
    model = LPNGeoLite(
        emb_size=4,
        backbone=shared,
        cross_attention=IdentityCrossAttention(),
        projection_dim=4,
        num_parts=2,
    )
    output = model(torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8))

    assert_query_guard_contract(output)
    assert output["query_native"].shape == (2, 8)


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"passed {len(tests)} tests")
