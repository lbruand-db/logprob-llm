# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Demo: calibrated typed routing vs a frontier chat call
# MAGIC Our GPU pyfunc returns `{value, team, confidence}` in one shot; a frontier
# MAGIC chat model returns prose we must parse and has no calibrated probability
# MAGIC (SPECS §11, build-plan WI-6).

# COMMAND ----------
# MAGIC %pip install -q "databricks-sdk>=0.102.0" openai numpy scipy
# MAGIC %restart_python

# COMMAND ----------
import os
import re
import sys
import time
sys.path.insert(0, os.path.abspath("../src"))
from logprob_llm.config import DEFAULT_ENDPOINT, DEFAULT_TEACHER_MODEL, _TEACHER_HELP
from logprob_llm.labels import LETTERS
from logprob_llm.prompts import build_routing_prompt
from logprob_llm.serving_read import make_client, route

dbutils.widgets.text("endpoint", DEFAULT_ENDPOINT, "Our serving endpoint")
dbutils.widgets.text("teacher_model", DEFAULT_TEACHER_MODEL, _TEACHER_HELP)
endpoint = dbutils.widgets.get("endpoint"); teacher = dbutils.widgets.get("teacher_model")

w = make_client()

# OpenAI-compatible client for the frontier baseline.
from openai import OpenAI
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
host = f"https://{ctx.tags().get('browserHostName').get()}"
oai = OpenAI(base_url=f"{host}/serving-endpoints", api_key=ctx.apiToken().get())

# COMMAND ----------
SAMPLES = [
    "I can't log in — password reset emails never arrive.",
    "Your dashboard shows the wrong totals after the latest update; looks like a bug.",
    "We're evaluating your Enterprise tier for 200 seats — can someone walk us through pricing?",
    "I was billed twice this month, please refund the duplicate charge.",
]

for s in SAMPLES:
    t0 = time.time(); out = route(w, endpoint, s); dt = (time.time() - t0) * 1000
    print(f"\nTICKET: {s}")
    print(f"  OURS  -> {out['value']} ({out['team']})  conf={out['confidence']:.2f}  [{dt:.0f} ms]")

# COMMAND ----------
# Frontier chat baseline: returns prose; parse the first valid letter.
def teacher_route(text):
    r = oai.chat.completions.create(
        model=teacher,
        messages=[{"role": "user", "content": build_routing_prompt(text)}],
        max_tokens=32, temperature=0.0)
    raw = (r.choices[0].message.content or "").strip()
    m = re.search(f"[{''.join(LETTERS)}]", raw)
    return (m.group(0) if m else "?"), raw

for s in SAMPLES[:2]:
    t0 = time.time(); letter, raw = teacher_route(s); dt = (time.time() - t0) * 1000
    print(f"TICKET: {s}\n  TEACHER -> {letter}  (raw: {raw!r})  [{dt:.0f} ms]  (no calibrated probability)")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Cost snippet
# MAGIC Ours emits one forward pass with a calibrated distribution; the frontier
# MAGIC baseline pays for input every call and returns prose. Plug the observed
# MAGIC latency/throughput into the SPECS §11 model (GPU DBU/hr × $/DBU vs $/1M
# MAGIC tokens) for the customer's volume to get the break-even.
