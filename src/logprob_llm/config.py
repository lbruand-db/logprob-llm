"""Shared defaults, so notebooks don't drift from one another.

Endpoint names are workspace/region/entitlement-dependent — always confirm the
chosen teacher exists via `w.serving_endpoints.list()` before a bulk run
(build-plan preflight #4).
"""

from __future__ import annotations

# Teacher used for label-first data generation (notebook 01) and the demo
# baseline (notebook 05). Frontier-class at moderate cost; good for bulk gen.
DEFAULT_TEACHER_MODEL = "databricks-claude-sonnet-5"

# Valid alternatives (full Databricks Foundation Model endpoint IDs):
TEACHER_ALTERNATIVES = {
    "max_quality": "databricks-claude-opus-5",
    "cheapest_strong": "databricks-llama-4-maverick",
    "open_reasoning": "databricks-gpt-oss-120b",
}

_TEACHER_HELP = (
    "Teacher FM endpoint. Alternatives: databricks-claude-opus-5 (max quality), "
    "databricks-llama-4-maverick (cheapest strong)."
)

# --- Milestone 1 run defaults (wired to fe-vm-serverless-stable-hma18t; edit per workspace) ---
# UC catalog you can CREATE schemas/volumes in (this FEVM: not 'main').
DEFAULT_CATALOG = "serverless_stable_hma18t_catalog"
DEFAULT_SCHEMA = "logprob"
DEFAULT_VOLUME = "data"

# Base model, fine-tuned via AI Runtime (Serverless GPU + TRL/LoRA). Qwen is
# supported there (unlike the older Foundation Model Fine-tuning API, which is
# Llama-only). Drop to Qwen/Qwen3-0.6B for the cheapest option.
DEFAULT_BASE_MODEL = "Qwen/Qwen3-1.7B"

# UC model + GPU serving endpoint names for the fine-tuned router pyfunc.
DEFAULT_REGISTERED_NAME = "logprob_qwen3"
DEFAULT_ENDPOINT = "logprob-qwen3"
