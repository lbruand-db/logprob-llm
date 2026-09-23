"""Label-first synthetic ticket generation (SPECS §5.1, build-plan WI-2).

We sample the true team label first, then ask a teacher model to write a
realistic ticket for it — so the seed label is gold, for free. Output rows are
``{"prompt": <routing prompt>, "response": " X", "label": "X", "ticket": ...}``;
the fine-tune uses prompt+response, calibration/eval use label.

Teacher access is via any OpenAI-compatible client — e.g. a Databricks
Foundation Model API endpoint (`{host}/serving-endpoints`) so generation stays
on-platform. Import `build_dataset` from a notebook, or run as a CLI.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from typing import Any

from logprob_llm.labels import TEAMS, answer_token_strings
from logprob_llm.prompts import build_generation_prompt, build_routing_prompt

_ANSWER = answer_token_strings()


def _parse_tickets(text: str) -> list[str]:
    """Split a teacher response into individual tickets.

    The teacher is asked for one ticket per call by default, but tolerate
    numbered/blank-line-separated lists too.
    """
    chunks = re.split(r"\n\s*\n|\n\d+[\.\)]\s+", text.strip())
    out = []
    for c in chunks:
        c = re.sub(r"^\s*\d+[\.\)]\s*", "", c).strip().strip('"')
        if len(c) >= 20:
            out.append(c)
    return out or ([text.strip()] if text.strip() else [])


def _one_ticket(client: Any, model: str, letter: str, name: str, temperature: float) -> str | None:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": build_generation_prompt(letter, name)}],
        temperature=temperature,
        max_tokens=200,
    )
    tickets = _parse_tickets(resp.choices[0].message.content or "")
    return tickets[0] if tickets else None


def build_dataset(
    client: Any,
    teacher_model: str,
    n_per_team: int = 200,
    temperature: float = 1.0,
    seed: int = 0,
) -> list[dict[str, str]]:
    """Generate a balanced, label-first dataset. Returns a shuffled list of rows."""
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for letter, name in TEAMS:
        made = 0
        attempts = 0
        while made < n_per_team and attempts < n_per_team * 3:
            attempts += 1
            ticket = _one_ticket(client, teacher_model, letter, name, temperature)
            if not ticket:
                continue
            rows.append(
                {
                    "ticket": ticket,
                    "label": letter,
                    "prompt": build_routing_prompt(ticket),
                    "response": _ANSWER[letter],
                }
            )
            made += 1
    rng.shuffle(rows)
    return rows


def split(rows: list[dict[str, str]], eval_frac: float = 0.1, calib_frac: float = 0.1):
    """Chronological split into (train, eval, calib). Input should be pre-shuffled."""
    n = len(rows)
    n_eval = int(n * eval_frac)
    n_calib = int(n * calib_frac)
    calib = rows[:n_calib]
    ev = rows[n_calib : n_calib + n_eval]
    train = rows[n_calib + n_eval :]
    return train, ev, calib


def write_jsonl(rows: list[dict[str, str]], path: str, fields: tuple[str, ...]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in fields}) + "\n")


def _cli() -> None:
    import os

    from openai import OpenAI

    ap = argparse.ArgumentParser(description="Generate synthetic ticket-routing data.")
    ap.add_argument("--teacher-model", required=True, help="Teacher endpoint/model name")
    ap.add_argument("--n-per-team", type=int, default=200)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY") or os.environ.get("DATABRICKS_TOKEN"))
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    rows = build_dataset(client, args.teacher_model, n_per_team=args.n_per_team)
    train, ev, calib = split(rows)
    write_jsonl(train, f"{args.out_dir}/train.jsonl", ("prompt", "response"))
    write_jsonl(ev, f"{args.out_dir}/eval.jsonl", ("prompt", "response", "label"))
    write_jsonl(calib, f"{args.out_dir}/calib.jsonl", ("prompt", "response", "label"))
    print(f"train={len(train)} eval={len(ev)} calib={len(calib)} -> {args.out_dir}")


if __name__ == "__main__":
    _cli()
