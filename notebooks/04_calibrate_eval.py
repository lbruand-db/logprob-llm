# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Calibrate & evaluate  (Milestone 1 proof point)
# MAGIC Pulls the T-independent `answer_logits` from the served pyfunc, fits the
# MAGIC temperature T on the calibration split, and reports accuracy + ECE / Brier /
# MAGIC log-loss **before vs after** with a reliability diagram (SPECS §8–§9, WI-5).
# MAGIC Acceptance: ECE drops after temperature scaling, accuracy unchanged.

# COMMAND ----------
# MAGIC %pip install -q "databricks-sdk>=0.102.0" numpy scipy matplotlib
# MAGIC %restart_python

# COMMAND ----------
import json
import os
import sys
sys.path.insert(0, os.path.abspath("../src"))
import numpy as np
from logprob_llm.config import DEFAULT_CATALOG, DEFAULT_ENDPOINT, DEFAULT_SCHEMA, DEFAULT_VOLUME

dbutils.widgets.text("catalog", DEFAULT_CATALOG, "UC catalog")
dbutils.widgets.text("schema", DEFAULT_SCHEMA, "Schema")
dbutils.widgets.text("volume", DEFAULT_VOLUME, "Volume")
dbutils.widgets.text("endpoint", DEFAULT_ENDPOINT, "Serving endpoint")

catalog = dbutils.widgets.get("catalog"); schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume"); endpoint = dbutils.widgets.get("endpoint")
vol = f"/Volumes/{catalog}/{schema}/{volume}"

from logprob_llm import calibrate
from logprob_llm.labels import LETTERS
from logprob_llm.serving_read import collect_logits, make_client

LETTER_IX = {ltr: i for i, ltr in enumerate(LETTERS)}

def load(split):
    with open(f"{vol}/{split}.jsonl") as f:
        rows = [json.loads(x) for x in f]
    return [r["ticket"] for r in rows], np.array([LETTER_IX[r["label"]] for r in rows])

# COMMAND ----------
client = make_client()
calib_tickets, calib_y = load("calib")
eval_tickets, eval_y = load("eval")
calib_logits = np.array(collect_logits(client, endpoint, calib_tickets))
eval_logits = np.array(collect_logits(client, endpoint, eval_tickets))

# COMMAND ----------
T = calibrate.fit_temperature(calib_logits, calib_y)
report = calibrate.calibration_report(eval_logits, eval_y, T)
print("Fitted temperature T =", round(T, 3))
for k, v in report.items():
    print(f"  {k:>16}: {v:.4f}")

# Persist T: a JSON artifact for the pyfunc (re-run 03 with artifacts["calibration"]
# = this file to apply T at serve time) and a UC table for the record.
with open(f"{vol}/calibration.json", "w") as f:
    json.dump({"temperature": T, "endpoint": endpoint}, f)
spark.sql(f"CREATE TABLE IF NOT EXISTS {catalog}.{schema}.calibration "
          "(endpoint STRING, temperature DOUBLE, ece_before DOUBLE, ece_after DOUBLE)")
spark.sql(f"DELETE FROM {catalog}.{schema}.calibration WHERE endpoint = '{endpoint}'")
spark.sql(f"INSERT INTO {catalog}.{schema}.calibration VALUES "
          f"('{endpoint}', {T}, {report['ece_before']}, {report['ece_after']})")

# COMMAND ----------
# Reliability diagram: raw vs calibrated.
import matplotlib.pyplot as plt

def bin_stats(probs, y, n=10):
    conf = probs.max(1); pred = probs.argmax(1); corr = (pred == y).astype(float)
    edges = np.linspace(0, 1, n + 1); xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            xs.append(conf[m].mean()); ys.append(corr[m].mean())
    return xs, ys

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot([0, 1], [0, 1], "k--", label="perfect")
for T_, lbl in [(1.0, f"raw (ECE {report['ece_before']:.3f})"),
                (T, f"calibrated (ECE {report['ece_after']:.3f})")]:
    xs, ys = bin_stats(calibrate.softmax(eval_logits, T_), eval_y)
    ax.plot(xs, ys, marker="o", label=lbl)
ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.legend(); ax.set_title("Reliability")
display(fig)
