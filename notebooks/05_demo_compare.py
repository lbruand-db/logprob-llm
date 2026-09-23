# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Demo: calibrated one-token routing vs a frontier chat call
# MAGIC Side-by-side on sample tickets, plus a cost snippet parameterized to the
# MAGIC live endpoint (SPECS §11, build-plan WI-6).

# COMMAND ----------
# MAGIC %pip install -q openai numpy scipy
# MAGIC %restart_python

# COMMAND ----------
import re
import sys, os, time
sys.path.insert(0, os.path.abspath("../src"))
from logprob_llm.config import DEFAULT_TEACHER_MODEL, _TEACHER_HELP

dbutils.widgets.text("catalog", "main", "UC catalog")
dbutils.widgets.text("schema", "logprob", "Schema")
dbutils.widgets.text("endpoint", "logprob-qwen3", "Our serving endpoint")
dbutils.widgets.text("teacher_model", DEFAULT_TEACHER_MODEL, _TEACHER_HELP)

catalog = dbutils.widgets.get("catalog"); schema = dbutils.widgets.get("schema")
endpoint = dbutils.widgets.get("endpoint"); teacher = dbutils.widgets.get("teacher_model")

from logprob_llm.serving_read import make_client, choice
from logprob_llm.prompts import build_routing_prompt

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
client = make_client(f"https://{ctx.tags().get('browserHostName').get()}", ctx.apiToken().get())

# Load calibration temperature persisted in notebook 04.
T = spark.sql(f"SELECT temperature FROM {catalog}.{schema}.calibration WHERE endpoint='{endpoint}'").collect()[0][0]
print("Using calibration T =", round(T, 3))

# COMMAND ----------
SAMPLES = [
    "I can't log in — password reset emails never arrive.",
    "Your dashboard shows the wrong totals after the latest update; looks like a bug.",
    "We're evaluating your Enterprise tier for 200 seats — can someone walk us through pricing?",
    "I was billed twice this month, please refund the duplicate charge.",
]

for s in SAMPLES:
    t0 = time.time()
    ours = choice(client, endpoint, s, temperature=T)
    dt = (time.time() - t0) * 1000
    print(f"\nTICKET: {s}")
    print(f"  OURS  -> {ours['value']} ({ours['team']})  conf={ours['confidence']:.2f}  [{dt:.0f} ms]")

# COMMAND ----------
# Frontier chat baseline for contrast (returns prose; we parse the letter).
# A frontier model may emit whitespace/preamble/reasoning before the letter, so
# give it room and extract the first valid routing letter robustly.
from logprob_llm.labels import LETTERS

def teacher_route(text):
    r = client.chat.completions.create(
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
# MAGIC Ours emits **one token** with a calibrated distribution; the frontier baseline
# MAGIC pays for input every call and returns prose. Plug the observed latency/throughput
# MAGIC into the SPECS §11 model (parameterized DBU/hr × $/DBU vs $/1M tokens) for the
# MAGIC customer's volume to get the break-even.
