# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Prepare data (label-first synthetic tickets)
# MAGIC Generates balanced ticket-routing data with a teacher model and writes
# MAGIC `train/eval/calib.jsonl` to the UC volume (SPECS §5.1, build-plan WI-2).

# COMMAND ----------
# MAGIC %pip install -q openai numpy scipy
# MAGIC %restart_python

# COMMAND ----------
import sys, os
sys.path.insert(0, os.path.abspath("../src"))
sys.path.insert(0, os.path.abspath("../data"))

dbutils.widgets.text("catalog", "main", "UC catalog")
dbutils.widgets.text("schema", "logprob", "Schema")
dbutils.widgets.text("volume", "data", "Volume")
dbutils.widgets.text("teacher_model", "databricks-meta-llama-3-3-70b-instruct", "Teacher FM endpoint")
dbutils.widgets.text("n_per_team", "300", "Tickets per team")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")
teacher = dbutils.widgets.get("teacher_model")
n_per_team = int(dbutils.widgets.get("n_per_team"))
vol = f"/Volumes/{catalog}/{schema}/{volume}"

# COMMAND ----------
# OpenAI-compatible client pointed at this workspace's Foundation Model APIs.
from openai import OpenAI
from generate_dataset import build_dataset, split, write_jsonl

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
host = f"https://{ctx.tags().get('browserHostName').get()}"
token = ctx.apiToken().get()
client = OpenAI(base_url=f"{host}/serving-endpoints", api_key=token)

# COMMAND ----------
rows = build_dataset(client, teacher, n_per_team=n_per_team)
train, ev, calib = split(rows)
write_jsonl(train, f"{vol}/train.jsonl", ("prompt", "response"))
write_jsonl(ev,    f"{vol}/eval.jsonl",  ("prompt", "response", "label"))
write_jsonl(calib, f"{vol}/calib.jsonl", ("prompt", "response", "label"))
print(f"train={len(train)} eval={len(ev)} calib={len(calib)}  ->  {vol}")

# COMMAND ----------
# Peek at a few rows to sanity-check prompt/response shape.
for r in train[:3]:
    print(r["response"], "|", r["prompt"][-120:].replace("\n", " "))
