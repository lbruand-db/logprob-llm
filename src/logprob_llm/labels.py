"""The closed answer space for the ticket-routing `Choice` primitive.

Each team maps to a single-letter answer. The model is fine-tuned so its answer
is exactly one of these letters, and at inference we read the log-probabilities
of the corresponding answer *tokens* (SPECS §4, §6.1). For that read to be a
single forward pass, every answer must be **one token** in the base tokenizer —
`resolve_answer_tokens` verifies this and picks the representation that works.
"""

from __future__ import annotations

from typing import Protocol


# Ordered source of truth. The letter is the answer symbol; the name is only for
# prompts / human-readable output. Keep this list == the options shown in prompts.py.
TEAMS: list[tuple[str, str]] = [
    ("A", "Billing & Payments"),
    ("B", "Technical Support"),
    ("C", "Account & Login"),
    ("D", "Sales & Pre-Sales"),
    ("E", "Bug Report"),
    ("F", "Feedback & Other"),
]

LETTERS: list[str] = [letter for letter, _ in TEAMS]
TEAM_NAME: dict[str, str] = {letter: name for letter, name in TEAMS}


class Tokenizer(Protocol):
    """Minimal interface satisfied by a HuggingFace tokenizer."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...


def resolve_answer_tokens(tokenizer: Tokenizer) -> dict[str, int]:
    """Map each letter to the single token id the model must emit.

    Tries the bare letter first (``"A"``), then the space-prefixed form
    (``" A"``) — many BPE tokenizers make the leading-space form the natural
    single token after ``"Answer:"``. Raises if neither is a single token, which
    is a hard blocker: the whole design depends on a one-token answer.
    """
    resolved: dict[str, int] = {}
    problems: list[str] = []
    for letter in LETTERS:
        chosen: int | None = None
        for candidate in (letter, f" {letter}"):
            ids = tokenizer.encode(candidate, add_special_tokens=False)
            if len(ids) == 1:
                chosen = ids[0]
                break
        if chosen is None:
            problems.append(letter)
        else:
            resolved[letter] = chosen
    if problems:
        raise ValueError(
            "These labels do not tokenize to a single token in this tokenizer "
            f"(as bare or space-prefixed): {problems}. Pick different answer "
            "symbols or a different base model (SPECS §12, tokenization note)."
        )
    if len(set(resolved.values())) != len(resolved):
        raise ValueError(f"Answer tokens collide (non-distinct ids): {resolved}")
    return resolved


def answer_token_strings() -> dict[str, str]:
    """Canonical response string per letter for the training JSONL.

    We emit the space-prefixed form because the prompt ends with ``"Answer:"``
    and the model's first generated token is typically space-prefixed. If a
    tokenizer prefers the bare form, `resolve_answer_tokens` still handles it at
    read time; training just needs a consistent target.
    """
    return {letter: f" {letter}" for letter in LETTERS}
