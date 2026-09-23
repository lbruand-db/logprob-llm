# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Unity Catalog setup
# MAGIC Creates the schema + volume that hold the dataset and the fine-tuned model.
# MAGIC Run once. Requires CREATE on the target catalog (build-plan preflight #1).

# COMMAND ----------
import os
import sys
sys.path.insert(0, os.path.abspath("../src"))
from logprob_llm.config import DEFAULT_CATALOG, DEFAULT_SCHEMA, DEFAULT_VOLUME

dbutils.widgets.text("catalog", DEFAULT_CATALOG, "UC catalog (writable)")
dbutils.widgets.text("schema", DEFAULT_SCHEMA, "Schema to create")
dbutils.widgets.text("volume", DEFAULT_VOLUME, "Volume for datasets + model")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")

# COMMAND ----------
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume}")

vol_path = f"/Volumes/{catalog}/{schema}/{volume}"
print("Schema :", f"{catalog}.{schema}")
print("Volume :", vol_path)
dbutils.fs.mkdirs(vol_path)

# COMMAND ----------
# MAGIC %md Next: **01_prepare_data** writes train/eval/calib JSONL into this volume.
