"""The typed serving surface: an MLflow pyfunc that returns a calibrated
distribution over the answer letters, not prose (SPECS §7, §12.1-C).

Custom LLM serving is `llm/v1/chat`, which does not reliably expose token
logprobs — so instead of reading endpoint logprobs we do the log-prob math
*inside* the served model: one forward pass, take the next-token logits at the
answer position over the answer-letter token ids, temperature-scale, and return
`{value, team, probabilities, confidence, answer_logits}`.

The scoring core (`select_answer_logits`, `score`) is pure numpy so it is unit
tested without torch/transformers/mlflow. The `LogProbRouter` PythonModel is
only defined when mlflow is importable (i.e. in the serving/training image).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .calibrate import softmax
from .labels import LETTERS, TEAM_NAME
from .prompts import build_routing_prompt

_MISSING_LOGIT = -30.0


def select_answer_logits(vocab_logits: np.ndarray, answer_ids: dict[str, int]) -> list[float]:
    """Gather the next-token logits for each answer letter, aligned to LETTERS.

    ``vocab_logits`` is the (vocab,) next-token logit vector; ``answer_ids`` maps
    each letter to its single token id (from `labels.resolve_answer_tokens`).
    """
    vocab_logits = np.asarray(vocab_logits, dtype=np.float64)
    out: list[float] = []
    for letter in LETTERS:
        tid = answer_ids.get(letter)
        out.append(float(vocab_logits[tid]) if tid is not None else _MISSING_LOGIT)
    return out


def score(answer_logits: list[float], temperature: float = 1.0) -> dict[str, Any]:
    """Typed contract from per-letter logits: {value, team, probabilities, confidence}.

    ``answer_logits`` is aligned to LETTERS. ``temperature`` is the calibration T
    (SPECS §8); T only reshapes confidence, never the argmax.
    """
    probs = softmax(np.array([answer_logits], dtype=np.float64), temperature)[0]
    dist = {LETTERS[i]: float(probs[i]) for i in range(len(LETTERS))}
    value = max(dist, key=dist.get)
    return {
        "value": value,
        "team": TEAM_NAME[value],
        "probabilities": dist,
        "confidence": dist[value],
        "answer_logits": [float(x) for x in answer_logits],  # T-independent, for calibration
    }


try:
    import mlflow.pyfunc

    class LogProbRouter(mlflow.pyfunc.PythonModel):
        """Serves the fine-tuned Qwen router as a calibrated typed classifier.

        Artifacts: ``model`` (a directory with the merged fine-tuned model +
        tokenizer, e.g. copied from a UC Volume) and optional ``calibration`` (a
        JSON file ``{"temperature": T}``). Input: a pandas DataFrame or list with
        a ``ticket`` column/field. Output: one `score(...)` dict per row.
        """

        def load_context(self, context):  # noqa: D102 - mlflow hook
            import json

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            from .labels import resolve_answer_tokens

            model_path = context.artifacts["model"]
            self._torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto"
            ).eval()
            self.answer_ids = resolve_answer_tokens(self.tokenizer)
            self.temperature = 1.0
            calib = context.artifacts.get("calibration") if context.artifacts else None
            if calib:
                with open(calib) as f:
                    self.temperature = float(json.load(f).get("temperature", 1.0))

        def _tickets(self, model_input) -> list[str]:
            if hasattr(model_input, "to_dict"):  # pandas DataFrame
                col = "ticket" if "ticket" in model_input.columns else model_input.columns[0]
                return [str(x) for x in model_input[col].tolist()]
            if isinstance(model_input, dict):
                return [str(x) for x in model_input.get("ticket", [])]
            return [str(x) for x in model_input]

        def predict(self, context, model_input, params=None):  # noqa: D102 - mlflow hook
            torch = self._torch
            results = []
            for ticket in self._tickets(model_input):
                prompt = build_routing_prompt(ticket)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    logits = self.model(**inputs).logits[0, -1, :].float().cpu().numpy()
                answer_logits = select_answer_logits(logits, self.answer_ids)
                results.append(score(answer_logits, self.temperature))
            return results

except ImportError:  # mlflow not installed (local test env); scoring core still importable
    LogProbRouter = None  # type: ignore[assignment,misc]
