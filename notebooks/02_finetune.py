# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Fine-tune with AI Runtime (Mosaic AI Model Training)
# MAGIC INSTRUCTION_FINETUNE of a small Qwen3 on the routing data, registered to UC
# MAGIC (SPECS §5.2, build-plan WI-3). Confirm the exact model string against the
# MAGIC workspace's supported-models list before running (preflight #2).

# COMMAND ----------
# MAGIC %pip install -q databricks-model-training
# MAGIC %restart_python

# COMMAND ----------
dbutils.widgets.text("catalog", "main", "UC catalog")
dbutils.widgets.text("schema", "logprob", "Schema")
dbutils.widgets.text("volume", "data", "Volume")
dbutils.widgets.text("base_model", "Qwen/Qwen3-1.7B", "Base model (or Qwen3-0.6B)")
dbutils.widgets.text("registered_name", "logprob_qwen3", "UC model name")
dbutils.widgets.text("training_duration", "3ep", "Epochs")

catalog = dbutils.widgets.get("catalog"); schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume"); base_model = dbutils.widgets.get("base_model")
reg = f"{catalog}.{schema}." + dbutils.widgets.get("registered_name")
vol = f"/Volumes/{catalog}/{schema}/{volume}"

# COMMAND ----------
from databricks.model_training import foundation_model as fm

run = fm.create(
    model=base_model,
    task_type="INSTRUCTION_FINETUNE",
    train_data_path=f"{vol}/train.jsonl",
    eval_data_path=f"{vol}/eval.jsonl",
    register_to=reg,
    training_duration=dbutils.widgets.get("training_duration"),
    learning_rate="5e-6",
)
print("Run:", run.name)

# COMMAND ----------
# Stream status until the run finishes and the model is registered in UC.
import time
from databricks.model_training import foundation_model as fm
while True:
    r = fm.get(run.name)
    print(r.status)
    if str(r.status) in ("COMPLETED", "FAILED", "STOPPED"):
        break
    time.sleep(60)
print("Registered model:", reg)
