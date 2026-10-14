"""Unit tests for Modules 4 & 5 (src/models.py).

Verifies 8 target behaviors:
1. 1D-CNN + 2-layer Bi-LSTM forward pass output shape (Batch, 28)
2. Sequence window dimension handling (T=30, D=1629)
3. Dynamic Classifier Head (LayerNorm, Dropout, Linear projection)
4. Softmax confidence calibration with temperature parameter
5. Model weight save and load checkpoint integrity
6. predict_window() output token and confidence bounds [0.0, 1.0]
7. Backward gradient computation for training optimization
8. Batched sequence inference (B=4, T=30, D=1629)
"""

import os
import tempfile
import torch
import numpy as np
import pytest
from src.models import SpatioTemporalSequenceModel


def test_model_forward_pass_shape():
    """Test 1: Forward pass returns logits tensor of shape (1, 28)."""
    model = SpatioTemporalSequenceModel(in_features=1629, num_classes=28)
    dummy_input = torch.randn(1, 30, 1629)
    logits = model(dummy_input)
    assert logits.shape == (1, 28)
    assert logits.dtype == torch.float32


def test_model_sequence_window_handling():
    """Test 2: Single 2D window (30, 1629) is automatically unsqueezed to batch."""
    model = SpatioTemporalSequenceModel(in_features=1629, num_classes=28)
    single_window = torch.randn(30, 1629)
    logits = model(single_window)
    assert logits.shape == (1, 28)


def test_model_classifier_head_components():
    """Test 3: Model contains LayerNorm, Dropout, and Linear projection head."""
    model = SpatioTemporalSequenceModel(in_features=1629, num_classes=28)
    assert hasattr(model, 'layer_norm')
    assert hasattr(model, 'dropout')
    assert hasattr(model, 'fc')
    assert model.fc.out_features == 28


def test_model_softmax_temperature_calibration():
    """Test 4: Temperature scaling controls output entropy/sharpness."""
    model_low_temp = SpatioTemporalSequenceModel(in_features=1629, num_classes=28, temperature=0.5)
    model_high_temp = SpatioTemporalSequenceModel(in_features=1629, num_classes=28, temperature=2.0)

    # Copy weights
    model_high_temp.load_state_dict(model_low_temp.state_dict())

    x = torch.randn(1, 30, 1629)
    _, conf_low = model_low_temp.predict_window(x)
    _, conf_high = model_high_temp.predict_window(x)

    # Lower temperature produces sharper/higher max confidence
    assert conf_low >= conf_high


def test_model_save_and_load_integrity():
    """Test 5: Model weights save and reload identically."""
    model = SpatioTemporalSequenceModel(in_features=1629, num_classes=28)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_model.pt")
        model.save(ckpt_path)
        assert os.path.exists(ckpt_path)

        model.eval()
        loaded_model = SpatioTemporalSequenceModel.load(ckpt_path, in_features=1629, num_classes=28)
        loaded_model.eval()
        x = torch.randn(1, 30, 1629)
        with torch.no_grad():
            orig_out = model(x)
            loaded_out = loaded_model(x)
        assert torch.allclose(orig_out, loaded_out, atol=1e-5)


def test_model_predict_window_bounds():
    """Test 6: predict_window() returns valid token and probability in [0, 1]."""
    vocab = ["SIGN_" + str(i) for i in range(28)]
    model = SpatioTemporalSequenceModel(in_features=1629, num_classes=28)
    window_data = np.random.randn(30, 1629).astype(np.float32)

    token, conf = model.predict_window(window_data, vocab)
    assert isinstance(token, str)
    assert token in vocab
    assert 0.0 <= conf <= 1.0


def test_model_gradient_backward_pass():
    """Test 7: Gradients flow cleanly through 1D-CNN, BiLSTM, and Classifier."""
    model = SpatioTemporalSequenceModel(in_features=1629, num_classes=28)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(2, 30, 1629)
    target = torch.tensor([0, 1], dtype=torch.long)

    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()

    assert model.conv1d.weight.grad is not None
    assert model.lstm.weight_ih_l0.grad is not None
    assert model.fc.weight.grad is not None
    optimizer.step()


def test_model_batched_sequence_inference():
    """Test 8: Model operates correctly on multi-batch tensors (Batch=4)."""
    model = SpatioTemporalSequenceModel(in_features=1629, num_classes=28)
    batch_input = torch.randn(4, 30, 1629)
    logits = model(batch_input)
    assert logits.shape == (4, 28)
