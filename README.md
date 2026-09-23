# logprob-llm

[![CI](https://github.com/lbruand-db/logprob-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/lbruand-db/logprob-llm/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
![Python](https://img.shields.io/badge/python-3.11%20|%203.12-blue)

An open-source **log-prob LLM** on Databricks: a small fine-tuned open-weight model
(default `Qwen/Qwen3-1.7B`) used as a *decision function* that returns a **calibrated
probability over a closed answer space** — a typed `{value, probability, confidence}` —
instead of free text.

- **Design:** [`SPEC/SPECS.md`](SPEC/SPECS.md)
- **Literature survey:** [`SPEC/SURVEY.md`](SPEC/SURVEY.md)

## Milestone 1 — support-ticket routing (the `Choice` primitive)

End-to-end thin slice: synthetic data → fine-tune via AI Runtime → serve on provisioned
throughput → read answer-token logprobs → temperature-calibrate → return the routed team
with a trustworthy confidence.

### Layout

| Path | Purpose |
|------|---------|
| `src/logprob_llm/labels.py` | Team label set + single-token answer verification |
| `src/logprob_llm/prompts.py` | The one prompt template used at train **and** inference |
| `src/logprob_llm/calibrate.py` | Temperature scaling + ECE / Brier / log-loss |
| `src/logprob_llm/serving_read.py` | Completions-endpoint logprob read → typed contract |
| `data/generate_dataset.py` | Label-first synthetic ticket generation |
| `notebooks/00…05` | Databricks: UC setup → data → fine-tune → serve → calibrate/eval → demo |
| `tests/` | Local, deterministic tests (calibration + label contract) |

### Local tests

```bash
uv run --with pytest --with numpy --with scipy pytest -q
```

### Databricks run

Authenticate (`fe-databricks-tools:databricks-authentication`), then run `notebooks/00`→`05`
in order. See `SPEC/SPECS.md` §5–§9 and the build plan for prerequisites (catalog, Model
Training + provisioned-throughput serving entitlements, a teacher model).
