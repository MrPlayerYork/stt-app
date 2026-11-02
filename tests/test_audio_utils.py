"""Tests for stt_app.audio.to_mono_16k (pure numpy, no audio hardware)."""
from __future__ import annotations

import numpy as np
import pytest

from stt_app.audio import TARGET_RATE, to_mono_16k


@pytest.mark.parametrize("rate", [48000, 16000])
def test_mono_output_shape_and_dtype(rate):
    n = rate  # one second
    x = np.sin(np.linspace(0, 20, n)).astype(np.float64)
    out = to_mono_16k(x, rate)
    assert out.dtype == np.float32
    assert out.ndim == 1
    assert abs(out.size - TARGET_RATE) <= 1


@pytest.mark.parametrize("rate", [48000, 16000])
def test_stereo_output_shape_and_dtype(rate):
    n = rate // 2  # half a second
    x = np.zeros((n, 2), dtype=np.float32)
    out = to_mono_16k(x, rate)
    assert out.dtype == np.float32
    assert out.ndim == 1
    assert abs(out.size - n * TARGET_RATE / rate) <= 1


def test_16k_mono_passthrough_length_and_values():
    x = np.linspace(-1, 1, 1600, dtype=np.float32)
    out = to_mono_16k(x, 16000)
    assert out.size == 1600
    np.testing.assert_allclose(out, x)


def test_48k_length_is_scaled():
    x = np.zeros(4800, dtype=np.float32)
    assert to_mono_16k(x, 48000).size == 1600


def test_stereo_is_averaged_to_mono():
    left = np.full(1600, 1.0, dtype=np.float32)
    right = np.full(1600, 0.0, dtype=np.float32)
    stereo = np.stack([left, right], axis=1)
    out = to_mono_16k(stereo, 16000)
    np.testing.assert_allclose(out, 0.5, atol=1e-6)


def test_stereo_mixes_channels_per_sample():
    stereo = np.array([[0.2, 0.4], [-1.0, 1.0], [0.6, 0.0]], dtype=np.float32)
    out = to_mono_16k(stereo, 16000)
    np.testing.assert_allclose(out, [0.3, 0.0, 0.3], atol=1e-6)


def test_empty_input_returns_empty_float32():
    out = to_mono_16k(np.zeros((0, 2), dtype=np.float32), 48000)
    assert out.dtype == np.float32
    assert out.size == 0
