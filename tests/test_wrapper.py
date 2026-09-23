"""Tests for the pure scoring core of the serving pyfunc (SPECS §7).

No torch/transformers/mlflow: we inject synthetic vocab logits and answer-token
ids, exactly what the served model computes internally.
"""

import numpy as np

from logprob_llm import labels, wrapper


def _answer_ids():
    # distinct token id per letter (as resolve_answer_tokens would return)
    return {letter: 1000 + i for i, letter in enumerate(labels.LETTERS)}


def _vocab_logits(peak_letter: str, peak: float = 8.0, vocab: int = 2000):
    ids = _answer_ids()
    v = np.full(vocab, -5.0)
    for tid in ids.values():
        v[tid] = 0.0
    v[ids[peak_letter]] = peak
    return v, ids


def test_select_answer_logits_aligns_to_letters():
    v, ids = _vocab_logits("C")
    picked = wrapper.select_answer_logits(v, ids)
    assert len(picked) == len(labels.LETTERS)
    # 'C' is the peak
    assert picked[labels.LETTERS.index("C")] == max(picked)


def test_select_handles_missing_letter():
    v, ids = _vocab_logits("A")
    ids.pop("F")  # simulate a letter with no resolved token
    picked = wrapper.select_answer_logits(v, ids)
    assert picked[labels.LETTERS.index("F")] == wrapper._MISSING_LOGIT


def test_score_contract_and_argmax():
    v, ids = _vocab_logits("D")
    out = wrapper.score(wrapper.select_answer_logits(v, ids))
    assert out["value"] == "D"
    assert out["team"] == labels.TEAM_NAME["D"]
    assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-9
    assert out["confidence"] == max(out["probabilities"].values())
    assert len(out["answer_logits"]) == len(labels.LETTERS)


def test_temperature_lowers_confidence_but_keeps_argmax():
    v, ids = _vocab_logits("B")
    logits = wrapper.select_answer_logits(v, ids)
    hot = wrapper.score(logits, temperature=1.0)
    cool = wrapper.score(logits, temperature=3.0)  # T>1 softens
    assert hot["value"] == cool["value"] == "B"
    assert cool["confidence"] < hot["confidence"]


def test_answer_logits_are_temperature_independent():
    v, ids = _vocab_logits("E")
    logits = wrapper.select_answer_logits(v, ids)
    a = wrapper.score(logits, temperature=1.0)["answer_logits"]
    b = wrapper.score(logits, temperature=5.0)["answer_logits"]
    assert a == b  # raw logits unchanged by T (used for calibration)
