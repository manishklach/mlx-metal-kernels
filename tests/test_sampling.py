import numpy as np
import pytest

from models import apply_repetition_penalty, greedy_sample, sample_logits, softmax, top_k_filter, top_p_filter


def test_softmax_sums_to_one():
    probs = np.asarray(softmax(np.array([1.0, 2.0, 3.0], dtype=np.float32)))
    assert np.allclose(probs.sum(), 1.0)


def test_greedy_sample_returns_argmax():
    assert greedy_sample(np.array([0.1, 2.5, 1.2], dtype=np.float32)) == 1


def test_top_k_filter_keeps_only_largest_k():
    filtered = np.asarray(top_k_filter(np.array([0.1, 2.0, 1.0, 3.0], dtype=np.float32), 2))
    kept = np.isfinite(filtered)
    assert kept.tolist() == [False, True, False, True]


def test_top_p_filter_keeps_probability_mass():
    logits = np.array([4.0, 3.0, 2.0, 0.1], dtype=np.float32)
    filtered = np.asarray(top_p_filter(logits, 0.8))
    assert np.isfinite(filtered).sum() >= 1
    assert np.isfinite(filtered).sum() < logits.shape[0]


def test_invalid_temperature_raises():
    with pytest.raises(ValueError, match="temperature must be positive"):
        sample_logits(np.array([1.0, 2.0]), temperature=0.0)


def test_invalid_top_k_raises():
    with pytest.raises(ValueError, match="top_k must be positive"):
        top_k_filter(np.array([1.0, 2.0]), 0)


def test_invalid_top_p_raises():
    with pytest.raises(ValueError, match="top_p must be in"):
        top_p_filter(np.array([1.0, 2.0]), 0.0)


def test_repetition_penalty_reduces_repeated_token_logits():
    logits = np.array([1.0, 3.0, -2.0], dtype=np.float32)
    penalized = np.asarray(apply_repetition_penalty(logits, [1, 2], penalty=1.5))
    assert penalized[1] < logits[1]
    assert penalized[2] < logits[2]


def test_sample_logits_returns_valid_token_id():
    logits = np.array([0.1, 2.0, 1.0], dtype=np.float32)
    token_id = sample_logits(logits, temperature=1.0, top_k=2, seed=9)
    assert token_id in {0, 1, 2}


def test_sample_logits_supports_batch():
    logits = np.array([[0.1, 2.0, 1.0], [1.5, 0.2, 0.3]], dtype=np.float32)
    token_ids = sample_logits(logits, temperature=1.0, top_p=0.9, seed=9)
    assert isinstance(token_ids, list)
    assert len(token_ids) == 2
