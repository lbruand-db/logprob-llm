"""Client for the typed router endpoint (SPECS §6).

The router is served as a custom MLflow pyfunc (see `wrapper.py`) that already
returns the calibrated typed contract, so this module just invokes the endpoint
and hands back its response. For calibration we also pull the raw, T-independent
`answer_logits` the pyfunc includes in each prediction.
"""

from __future__ import annotations

from typing import Any


def make_client() -> Any:
    """Databricks SDK WorkspaceClient (uses ambient auth / DATABRICKS_* env)."""
    from databricks.sdk import WorkspaceClient  # lazy: keep optional for local use

    return WorkspaceClient()


def _query(client: Any, endpoint: str, tickets: list[str]) -> list[dict[str, Any]]:
    """Invoke the pyfunc endpoint with one row per ticket; return prediction dicts."""
    resp = client.serving_endpoints.query(
        name=endpoint,
        dataframe_records=[{"ticket": t} for t in tickets],
    )
    preds = getattr(resp, "predictions", None)
    if preds is None and isinstance(resp, dict):
        preds = resp.get("predictions")
    return list(preds or [])


def route(client: Any, endpoint: str, ticket: str) -> dict[str, Any]:
    """Route one ticket; returns {value, team, probabilities, confidence, ...}."""
    preds = _query(client, endpoint, [ticket])
    return preds[0]


def collect_logits(client: Any, endpoint: str, tickets: list[str]) -> list[list[float]]:
    """Per-ticket raw answer-letter logits (T-independent) for calibration/eval.

    Returns an (N, K=len(LETTERS)) list; feed to `calibrate.fit_temperature`
    with gold label indices.
    """
    return [p["answer_logits"] for p in _query(client, endpoint, tickets)]
