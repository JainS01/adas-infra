"""Unit tests for FusionBaseline — architecture, shapes, parameter count."""

from __future__ import annotations

import torch

from adas_infra.train.models.fusion_baseline import FingerprintEncoder, FusionBaseline, IrisEncoder


class TestIrisEncoder:
    def test_output_shape(self):
        enc = IrisEncoder(embed_dim=128)
        x = torch.randn(4, 1, 64, 64)
        out = enc(x)
        assert out.shape == (4, 128)

    def test_handles_different_input_resolutions(self):
        enc = IrisEncoder(embed_dim=64)
        for h, w in [(32, 32), (64, 64), (128, 128)]:
            out = enc(torch.randn(2, 1, h, w))
            assert out.shape == (2, 64)


class TestFingerprintEncoder:
    def test_output_shape(self):
        enc = FingerprintEncoder(embed_dim=128)
        x = torch.randn(4, 1, 96, 96)
        out = enc(x)
        assert out.shape == (4, 128)


class TestFusionBaseline:
    def test_forward_output_shape(self):
        model = FusionBaseline(num_classes=20)
        iris = torch.randn(8, 1, 64, 64)
        fp = torch.randn(8, 1, 96, 96)
        logits = model(iris, fp)
        assert logits.shape == (8, 20)

    def test_num_classes_property(self):
        model = FusionBaseline(num_classes=50)
        assert model.num_classes == 50

    def test_parameter_count_reasonable(self):
        model = FusionBaseline(num_classes=20)
        params = sum(p.numel() for p in model.parameters())
        # Should be between 500K and 5M for this shallow architecture
        assert 200_000 < params < 5_000_000, f"Unexpected param count: {params}"

    def test_no_nan_in_forward(self):
        model = FusionBaseline(num_classes=10)
        iris = torch.randn(4, 1, 64, 64)
        fp = torch.randn(4, 1, 96, 96)
        logits = model(iris, fp)
        assert not torch.isnan(logits).any(), "NaN in forward pass output"

    def test_gradients_flow(self):
        model = FusionBaseline(num_classes=5)
        iris = torch.randn(2, 1, 64, 64, requires_grad=False)
        fp = torch.randn(2, 1, 96, 96, requires_grad=False)
        labels = torch.tensor([0, 1])
        logits = model(iris, fp)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_train_mode_handles_batch_size_one(self):
        # Regression: BatchNorm1d in the fusion head crashed on B=1 in train mode.
        # LayerNorm makes the head batch-size-independent.
        model = FusionBaseline(num_classes=5)
        model.train()
        iris = torch.randn(1, 1, 64, 64)
        fp = torch.randn(1, 1, 96, 96)
        logits = model(iris, fp)
        assert logits.shape == (1, 5)
        assert not torch.isnan(logits).any()

    def test_eval_mode_no_grad_same_output(self):
        model = FusionBaseline(num_classes=5)
        model.eval()
        iris = torch.randn(2, 1, 64, 64)
        fp = torch.randn(2, 1, 96, 96)
        with torch.no_grad():
            out1 = model(iris, fp)
            out2 = model(iris, fp)
        assert torch.allclose(out1, out2), "Non-deterministic eval output"
