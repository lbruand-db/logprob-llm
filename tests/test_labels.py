"""Tests for the label / answer-token contract (SPECS §4, §12 tokenization note).

Uses a fake tokenizer so the single-token logic is verified without installing
`transformers` or downloading a model. The real Qwen3 check runs in the
Databricks dev-loop where `transformers` is available.
"""

import pytest

from logprob_llm import labels


class FakeTokenizer:
    """Assigns ids from a fixed vocab; anything unknown splits per character.

    Configure which strings are single tokens via ``vocab``.
    """

    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab

    def encode(self, text: str, add_special_tokens: bool = False):
        if text in self.vocab:
            return [self.vocab[text]]
        # fall back to TWO ids per character, so anything not explicitly in the
        # vocab is multi-token (a realistic "this label isn't one token" case).
        ids: list[int] = []
        for c in text:
            ids.extend(divmod(ord(c), 128))
        return ids


def test_labels_and_teams_are_consistent():
    assert labels.LETTERS == [ltr for ltr, _ in labels.TEAMS]
    assert set(labels.TEAM_NAME) == set(labels.LETTERS)
    assert len(labels.LETTERS) == len(set(labels.LETTERS))  # distinct


def test_resolve_prefers_bare_single_token():
    vocab = {letter: i for i, letter in enumerate(labels.LETTERS)}
    tok = FakeTokenizer(vocab)
    resolved = labels.resolve_answer_tokens(tok)
    assert set(resolved) == set(labels.LETTERS)
    assert len(set(resolved.values())) == len(labels.LETTERS)  # distinct ids


def test_resolve_falls_back_to_space_prefixed():
    # bare letters are NOT single tokens; space-prefixed ARE
    vocab = {f" {letter}": 100 + i for i, letter in enumerate(labels.LETTERS)}
    tok = FakeTokenizer(vocab)
    resolved = labels.resolve_answer_tokens(tok)
    assert set(resolved) == set(labels.LETTERS)


def test_resolve_raises_when_not_single_token():
    # only some letters resolvable -> must raise, listing the offenders
    vocab = {"A": 1, "B": 2}  # C..F are not single tokens in either form
    tok = FakeTokenizer(vocab)
    with pytest.raises(ValueError, match="single token"):
        labels.resolve_answer_tokens(tok)


def test_resolve_raises_on_colliding_ids():
    vocab = {letter: 7 for letter in labels.LETTERS}  # all map to same id
    tok = FakeTokenizer(vocab)
    with pytest.raises(ValueError, match="collide"):
        labels.resolve_answer_tokens(tok)


def test_answer_token_strings_are_space_prefixed():
    strings = labels.answer_token_strings()
    assert strings["A"] == " A"
    assert set(strings) == set(labels.LETTERS)
