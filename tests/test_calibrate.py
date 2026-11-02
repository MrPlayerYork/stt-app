import numpy as np

from stt_app.audio import InputDevice
from stt_app.calibrate import MicScore, analyze_audio, combine, rank, score_transcript

T = "The quick brown fox jumps over the lazy dog."


def test_score_identical_and_case_punct():
    assert score_transcript(T, T) == 1.0
    assert score_transcript(T, "the QUICK brown fox, jumps over the lazy dog") == 1.0


def test_score_partial_and_empty():
    s = score_transcript(T, "the quick brown cat")
    assert 0.0 < s < 1.0
    assert score_transcript(T, "") == 0.0


def _sine(amp, n=16000):
    return (amp * np.sin(2 * np.pi * 440 * np.arange(n) / 16000)).astype(np.float32)


def test_analyze_silence_and_empty():
    assert analyze_audio(np.zeros(0, np.float32)) == (-120.0, 0.0, 0.0)
    assert analyze_audio(np.zeros(16000, np.float32)) == (-120.0, 0.0, 0.0)


def test_analyze_sine_levels():
    rms, _, clip = analyze_audio(_sine(0.5))
    assert -10 < rms < -8 and clip == 0.0
    assert analyze_audio(_sine(0.05))[0] < rms


def test_analyze_clipped_and_snr():
    assert analyze_audio(np.ones(16000, np.float32))[2] == 1.0
    x = np.concatenate([_sine(0.5), np.zeros(32000, np.float32) + 0.001])
    assert analyze_audio(x)[1] > 20


def test_combine_monotonic():
    assert combine(0.9, -20, 20, 0) > combine(0.5, -20, 20, 0)
    assert combine(0.9, -20, 30, 0) > combine(0.9, -20, 10, 0)
    assert combine(0.9, -20, 20, 0.2) < combine(0.9, -20, 20, 0)
    assert combine(0.9, -55, 20, 0) < combine(0.9, -20, 20, 0)


def _ms(i, score, err=None):
    d = InputDevice(i, f"d{i}", "JACK", 1, 48000.0)
    return MicScore(d, "", 0.0, -20.0, 10.0, 0.0, score, err)


def test_rank_errors_last():
    r = rank([_ms(0, 5), _ms(1, 99, "silent"), _ms(2, 50), _ms(3, -1e9, "x")])
    assert [s.device.index for s in r[:2]] == [2, 0]
    assert all(s.error for s in r[2:])
