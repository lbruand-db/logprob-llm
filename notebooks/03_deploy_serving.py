# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Deploy on Model Serving (provisioned throughput)
# MAGIC Serves the fine-tuned UC model and validates the **completions + logprobs**
# MAGIC read (SPECS §6, §6.1, build-plan WI-4).

# COMMAND ----------
# MAGIC %pip install -q databricks-sdk openai numpy scipy
# MAGIC %restart_python

# COMMAND ----------
import sys, os
sys.path.insert(0, os.path.abspath("../src"))

dbutils.widgets.text("catalog", "main", "UC catalog")
dbutils.widgets.text("schema", "logprob", "Schema")
dbutils.widgets.text("registered_name", "logprob_qwen3", "UC model name")
dbutils.widgets.text("model_version", "1", "Model version")
dbutils.widgets.text("endpoint", "logprob-qwen3", "Serving endpoint name")

catalog = dbutils.widgets.get("catalog"); schema = dbutils.widgets.get("schema")
entity = f"{catalog}.{schema}." + dbutils.widgets.get("registered_name")
version = dbutils.widgets.get("model_version")
endpoint = dbutils.widgets.get("endpoint")

# COMMAND ----------
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

w = WorkspaceClient()
w.serving_endpoints.create_and_wait(
    name=endpoint,
    config=EndpointCoreConfigInput(
        served_entities=[ServedEntityInput(
            entity_name=entity,
            entity_version=version,
            min_provisioned_throughput=0,        # scale-to-zero when idle
            max_provisioned_throughput=980,      # size to peak QPS after a load test
        )],
    ),
)
print("Endpoint READY:", endpoint)

# COMMAND ----------
# VALIDATION GATE (WI-4): does the completions endpoint return per-token logprobs?
from logprob_llm.serving_read import make_client, choice

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
host = f"https://{ctx.tags().get('browserHostName').get()}"
token = ctx.apiToken().get()
client = make_client(host, token)

sample = "I was charged twice for my subscription this month and need a refund."
try:
    out = choice(client, endpoint, sample, temperature=1.0)
    print("logprobs OK ->", out)
except Exception as e:
    print("GATE FAILED — completions logprobs unavailable; fall back to local "
          "scoring per SPECS §12.1-C. Error:", repr(e))
