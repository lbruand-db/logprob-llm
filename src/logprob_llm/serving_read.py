"""Read a calibrated distribution from a Databricks Model Serving endpoint.

Queries the OpenAI-compatible **completions** endpoint (logprobs are exposed
there, not on chat — SPECS §6.1) with a single forced token, then restricts to
the answer tokens and applies the calibration temperature T (SPECS §8).

Returns the typed contract ``{value, probabilities, confidence}``.
"""

from __future__ import annotations

import os
from typing import Any

from .calibrate import softmax
from .labels import LETTERS, TEAM_NAME
from .prompts import build_routing_prompt

# Very negative logit for an answer token the endpoint did not return in top_logprobs.
_MISSING_LOGIT = -30.0


def make_client(host: str | None = None, token: str | None = None) -> Any:
    """OpenAI client pointed at ``{host}/serving-endpoints``.

    Falls back to DATABRICKS_HOST / DATABRICKS_TOKEN env vars.
    """
    from openai import OpenAI  # lazy: keep import optional for non-serving use

    host = host or os.environ["DATABRICKS_HOST"]
    token = token or os.environ["DATABRICKS_TOKEN"]
    return OpenAI(base_url=f"{host.rstrip('/')}/serving-endpoints", api_key=token)


def _distribution_from_top_logprobs(
    top: dict[str, float], temperature: float
) -> dict[str, float]:
    """Turn the endpoint's {token_str: logprob} into a calibrated distribution.

    ``top`` is keyed by token *string*; we match on the string forms of our
    answer letters (bare and space-prefixed). Any answer letter absent from the
    returned top-k is assigned a floor logit before the temperature softmax.
    """
    import numpy as np

    logits = []
    for letter in LETTERS:
        candidates = (letter, f" {letter}")
        found = [top[c] for c in candidates if c in top]
        logits.append(max(found) if found else _MISSING_LOGIT)
    probs = softmax(np.array([logits]), temperature)[0]
    return {letter: float(probs[i]) for i, letter in enumerate(LETTERS)}


def choice(
    client: Any,
    endpoint: str,
    ticket_text: str,
    temperature: float = 1.0,
    top_logprobs: int = 20,
) -> dict[str, Any]:
    """Route one ticket; return {value, team, probabilities, confidence}.

    ``temperature`` is the calibration T from `calibrate.fit_temperature`
    (default 1.0 = uncalibrated raw softmax).
    """
    prompt = build_routing_prompt(ticket_text)
    resp = client.completions.create(
        model=endpoint,
        prompt=prompt,
        max_tokens=1,
        temperature=0.0,
        logprobs=True,
        top_logprobs=top_logprobs,
    )
    top = resp.choices[0].logprobs.top_logprobs[0]  # {token_str: logprob}
    dist = _distribution_from_top_logprobs(top, temperature)
    value = max(dist, key=dist.get)
    return {
        "value": value,
        "team": TEAM_NAME[value],
        "probabilities": dist,
        "confidence": dist[value],
    }


def collect_logits(
    client: Any,
    endpoint: str,
    tickets: list[str],
    top_logprobs: int = 20,
) -> "list[list[float]]":
    """Per-ticket raw logits over the answer letters, for calibration/eval.

    Returns an (N, K=len(LETTERS)) list; feed to `calibrate.fit_temperature`
    together with gold label indices.
    """
    rows: list[list[float]] = []
    for t in tickets:
        prompt = build_routing_prompt(t)
        resp = client.completions.create(
            model=endpoint, prompt=prompt, max_tokens=1, temperature=0.0,
            logprobs=True, top_logprobs=top_logprobs,
        )
        top = resp.choices[0].logprobs.top_logprobs[0]
        rows.append(
            [max([top[c] for c in (l, f" {l}") if c in top] or [_MISSING_LOGIT]) for l in LETTERS]
        )
    return rows
