# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Log the typed pyfunc & deploy GPU serving
# MAGIC Logs `LogProbRouter` (SPECS §7 / `src/logprob_llm/wrapper.py`) against the
# MAGIC merged model from notebook 02, registers it to UC, and creates a GPU
# MAGIC serving endpoint that returns `{value, team, probabilities, confidence,
# MAGIC answer_logits}` directly — no dependence on endpoint logprobs (build-plan v2 WI-4).

# COMMAND ----------
# MAGIC %pip install -q -U mlflow "databricks-sdk>=0.102.0" transformers torch accelerate numpy scipy
# MAGIC %restart_python

# COMMAND ----------
import os
import sys
sys.path.insert(0, os.path.abspath("../src"))
from logprob_llm.config import (
    DEFAULT_CATALOG,
    DEFAULT_ENDPOINT,
    DEFAULT_REGISTERED_NAME,
    DEFAULT_SCHEMA,
    DEFAULT_VOLUME,
)

dbutils.widgets.text("catalog", DEFAULT_CATALOG, "UC catalog")
dbutils.widgets.text("schema", DEFAULT_SCHEMA, "Schema")
dbutils.widgets.text("volume", DEFAULT_VOLUME, "Volume")
dbutils.widgets.text("registered_name", DEFAULT_REGISTERED_NAME, "UC pyfunc model name")
dbutils.widgets.text("endpoint", DEFAULT_ENDPOINT, "GPU serving endpoint name")

catalog = dbutils.widgets.get("catalog"); schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume"); endpoint = dbutils.widgets.get("endpoint")
name = dbutils.widgets.get("registered_name")
reg = f"{catalog}.{schema}.{name}"
merged_dir = f"/Volumes/{catalog}/{schema}/{volume}/models/{name}"

# COMMAND ----------
# Log the pyfunc. code_paths ships the logprob_llm package into the model so the
# wrapper's relative imports (labels/prompts/calibrate) resolve at serve time.
import mlflow
from logprob_llm.wrapper import LogProbRouter

mlflow.set_registry_uri("databricks-uc")
with mlflow.start_run(run_name="qwen3-router-pyfunc"):
    info = mlflow.pyfunc.log_model(
        name="router",
        python_model=LogProbRouter(),
        artifacts={"model": merged_dir},                 # + optional "calibration": <T.json> (notebook 04)
        code_paths=[os.path.abspath("../src/logprob_llm")],
        pip_requirements=["transformers", "torch", "accelerate", "numpy", "scipy", "mlflow"],
        registered_model_name=reg,
        input_example={"ticket": ["I was double charged this month."]},
    )
version = info.registered_model_version
print("registered pyfunc:", reg, "version", version)

# COMMAND ----------
# Create a GPU serving endpoint (A10 / GPU_MEDIUM) for the pyfunc.
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

w = WorkspaceClient()
w.serving_endpoints.create_and_wait(
    name=endpoint,
    config=EndpointCoreConfigInput(
        served_entities=[ServedEntityInput(
            entity_name=reg,
            entity_version=str(version),
            workload_type="GPU_MEDIUM",     # 1x A10; use GPU_XLARGE (H100) for bigger models
            workload_size="Small",
            scale_to_zero_enabled=True,
        )],
    ),
)
print("endpoint READY:", endpoint)

# COMMAND ----------
# Smoke test: the endpoint returns the typed contract directly.
from logprob_llm.serving_read import make_client, route

out = route(make_client(), endpoint, "I can't log in and the reset email never arrives.")
print(out)
