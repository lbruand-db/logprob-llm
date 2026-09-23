# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Fine-tune with AI Runtime (Serverless GPU + TRL LoRA)
# MAGIC Fine-tunes `Qwen/Qwen3-1.7B` on the routing data with TRL `SFTTrainer` +
# MAGIC LoRA, merges the adapter, and saves the merged model to a UC Volume for
# MAGIC serving in notebook 03 (SPECS §5, build-plan v2 WI-3).
# MAGIC
# MAGIC **Compute:** attach this notebook to **Serverless GPU** — Connect → Serverless
# MAGIC GPU → A10 (GPU_MEDIUM) → Environment **AI v6** → Apply. If you don't see
# MAGIC Serverless GPU, ask an admin to enable the AI Runtime preview.

# COMMAND ----------
# MAGIC %pip install -q -U trl peft transformers datasets accelerate mlflow "databricks-sdk>=0.102.0"
# MAGIC %restart_python

# COMMAND ----------
import os
import sys
sys.path.insert(0, os.path.abspath("../src"))
from logprob_llm.config import (
    DEFAULT_BASE_MODEL,
    DEFAULT_CATALOG,
    DEFAULT_REGISTERED_NAME,
    DEFAULT_SCHEMA,
    DEFAULT_VOLUME,
)

dbutils.widgets.text("catalog", DEFAULT_CATALOG, "UC catalog")
dbutils.widgets.text("schema", DEFAULT_SCHEMA, "Schema")
dbutils.widgets.text("volume", DEFAULT_VOLUME, "Volume")
dbutils.widgets.text("base_model", DEFAULT_BASE_MODEL, "Base model (Qwen3-0.6B for cheapest)")
dbutils.widgets.text("registered_name", DEFAULT_REGISTERED_NAME, "UC model name (merged HF model)")
dbutils.widgets.text("epochs", "3", "Epochs")

catalog = dbutils.widgets.get("catalog"); schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume"); base_model = dbutils.widgets.get("base_model")
epochs = float(dbutils.widgets.get("epochs"))
vol = f"/Volumes/{catalog}/{schema}/{volume}"
merged_dir = f"{vol}/models/{dbutils.widgets.get('registered_name')}"

# COMMAND ----------
# Verify the answer letters are single tokens in THIS tokenizer before training
# (the whole log-prob read depends on it — SPECS §12 tokenization note).
from transformers import AutoTokenizer
from logprob_llm.labels import resolve_answer_tokens

tok = AutoTokenizer.from_pretrained(base_model)
print("answer token ids:", resolve_answer_tokens(tok))  # raises if any label isn't 1 token

# COMMAND ----------
# Build the SFT dataset: text = prompt + response (response is the single answer
# letter, e.g. " C"). We train completion-only loss on the answer via TRL's
# response template so the model learns to emit the letter after "Answer:".
from datasets import load_dataset

ds = load_dataset("json", data_files=f"{vol}/train.jsonl", split="train")
ds = ds.map(lambda r: {"text": r["prompt"] + r["response"]})
print(ds[0]["text"][-160:])

# COMMAND ----------
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from trl.trainer import DataCollatorForCompletionOnlyLM

model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto")

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
# Loss only on tokens after "Answer:" (the letter), not the prompt.
collator = DataCollatorForCompletionOnlyLM(response_template="Answer:", tokenizer=tok)

trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    peft_config=lora,
    data_collator=collator,
    args=SFTConfig(
        output_dir=f"{vol}/ckpt/{dbutils.widgets.get('registered_name')}",
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        learning_rate=2e-4,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=25,
        dataset_text_field="text",
        max_seq_length=1024,
        report_to=[],
    ),
)
trainer.train()

# COMMAND ----------
# Merge the LoRA adapter into the base weights and save the merged model +
# tokenizer to the UC Volume (this is what the serving pyfunc loads in 03).
merged = trainer.model.merge_and_unload()
merged.save_pretrained(merged_dir)
tok.save_pretrained(merged_dir)
print("merged model saved to:", merged_dir)

# COMMAND ----------
# MAGIC %md
# MAGIC The merged model now lives in the UC Volume. **Next: notebook 03** logs the
# MAGIC `LogProbRouter` pyfunc against this directory and deploys a GPU serving endpoint.
# MAGIC (We don't register the raw HF model — the served artifact is our typed pyfunc.)
