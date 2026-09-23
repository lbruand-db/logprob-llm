# Spec: Fine-tune & deploy an OSS "log-prob" LLM (a Jev-like model) on Databricks

**Status:** Draft / demo blueprint
**Owner:** lbruand
**Last updated:** 2026-09-23
**Platform:** Databricks (Mosaic AI Model Training a.k.a. "AI Runtime" + Mosaic AI Model Serving + Unity Catalog)

---

## 0. TL;DR

Build the open-source analog of [**Jev**](https://en.wikipedia.org/wiki/Jev_(AI_model)) — an
LLM that, instead of generating prose, returns a **typed value together with a calibrated
probability**. We do it on Databricks by:

1. **Distilling** a frontier "teacher" model into a labeled/synthetic dataset in Unity Catalog.
2. **Fine-tuning** a small, *current* open-weight model (**Qwen3 1.7B / 0.6B** by default — see
   §3.1 for why, and the alternatives) with **AI Runtime / Mosaic AI Model Training** so the
   answer is always a **single, constrained token**.
3. **Reading the log-probabilities** of those answer tokens at inference (via the Model Serving
   `completions` endpoint) to recover a full probability distribution over the answer space.
4. **Calibrating** those probabilities (temperature scaling) so they mean what they say.
5. **Serving** a thin typed contract — `{value, probability, confidence}` — that mirrors Jev's
   `Choice` / `Score` / `Noul` primitives.

The payoff, like Jev's pitch, is **latency and cost**: a right-sized 1B model emitting one token
is far cheaper and faster than a frontier chat model reasoning in prose — and it runs inside the
customer's Databricks governance boundary. §11 quantifies the trade-off and the break-even.

> **Sources note.** The two links in the request were used as follows. The Wikipedia article on
> Jev was read and is the basis for §1. The `chatgpt.com/s/...` share link requires
> authentication and could **not** be fetched; if there is specific content in it we should
> incorporate, paste the transcript and this spec will be updated. All Databricks feature names,
> DBU rates, and third-party API prices below were gathered via web research on 2026-09-23 and
> are **illustrative** — verify against the live pricing pages and your contract before quoting a
> customer (see §13, References).

---

## 1. Background: what "Jev" is, and what "log-prob LLM" means

[Jev](https://en.wikipedia.org/wiki/Jev_(AI_model)) is a **proprietary** model from **TypeSafe
AI** (San Francisco, founded 2024; CEO Diogo Almeida, previously at OpenAI on RLHF / InstructGPT /
ChatGPT / GPT-4). It is deliberately **not** a chat model:

- **Output is for software, not humans.** Jev "returns typed values together with probability
  estimates and confidence scores." There is no free-text generation to parse, and therefore no
  hallucinated formatting or type errors.
- **Three question primitives:**
  - **Choice** — pick one of N options; returns a probability per option.
  - **Score** — an ordinal level (e.g. 1–5); returns a probability per level.
  - **Noul** — a single probability in `[0, 1]` (a calibrated yes/no likelihood).
- **Training:** transformer-based, trained *exclusively on synthetic data* with a method
  TypeSafe calls **RLCD (Reinforcement Learning for Calibrated Decisions)**, where
  "probabilities are optimized against **outcomes** rather than against human rater preference."
  Weights, parameter count and architecture are undisclosed.
- **Claims:** 70–500 ms responses; "40 to 200× faster and 40 to 400× cheaper than frontier LLMs"
  (self-reported, peak 193.6× faster / 444.6× cheaper). Limited early access launched
  2026-09-15; $40M seed led by DCVC.

**Why an OSS analog is feasible.** Every frontier and open model already computes a probability
distribution over the next token; a chat wrapper just samples from it and throws the numbers
away. If we (a) constrain the answer to a single known token per class and (b) expose the
per-token `logprobs`, we recover exactly the "typed value + probability" contract Jev sells — and
fine-tuning + post-hoc calibration is the pragmatic, reproducible stand-in for RLCD. This is a
well-established recipe (LLM-as-classifier + temperature scaling; Guo et al., 2017).

---

## 2. Goals & non-goals

**Goals**
- A reproducible Databricks demo that fine-tunes and serves a small OSS model returning
  **calibrated, typed** outputs with per-answer log-probabilities.
- Support the three Jev primitives (`Choice`, `Score`, `Noul`) behind one typed serving contract.
- A defensible **cost & latency comparison** vs. calling a frontier LLM API for the same task.
- Everything inside **Unity Catalog** governance (data, model, endpoint, lineage).

**Non-goals**
- Reproducing Jev's exact weights, RLCD algorithm, or benchmark numbers (undisclosed/proprietary).
- General-purpose text generation. This model does classification / scoring / probability only.
- Beating frontier models on hard open-ended reasoning — that is not the use case.

---

## 3. Architecture overview

```
                          ┌──────────────────────────── Databricks (Unity Catalog governed) ───────────────────────────┐
                          │                                                                                            │
  Labeled or unlabeled    │   ┌─────────────┐    ┌───────────────────────┐    ┌────────────────┐   ┌────────────────┐  │
  task data  ────────────►│   │ 1. Data prep │──► │ 2. AI Runtime          │──►│ UC model       │──►│ 4. Model       │  │
  (+ optional frontier    │   │  synthetic/  │    │  Model Training        │   │ registry        │   │  Serving (PT)  │  │
   teacher for distill.)  │   │  distilled   │    │  INSTRUCTION_FINETUNE  │   │ (fine-tuned)    │   │  completions   │  │
                          │   │  → JSONL     │    │  Qwen3-1.7B/0.6B        │   └────────────────┘   │  + logprobs    │  │
                          │   └─────────────┘    └───────────────────────┘                          └───────┬────────┘  │
                          │          │                                                                       │           │
                          │          ▼                                              logprobs over answer     ▼           │
                          │   ┌──────────────┐    ┌───────────────────────┐         tokens          ┌────────────────┐  │
                          │   │ held-out set │──► │ 3. Calibration          │◄─────────────────────── │ 5. Typed        │  │
                          │   │ (labels)     │    │  temperature scaling    │   store T in UC table   │ contract wrapper│  │
                          │   └──────────────┘    │  + ECE / Brier eval     │ ──────────────────────► │ {value, p, conf}│  │
                          │                       └───────────────────────┘                          └───────┬────────┘  │
                          └────────────────────────────────────────────────────────────────────────────────┼───────────┘
                                                                                                             ▼
                                                                                          Client apps / pipelines / Databricks Apps
```

**Base model choice.** Default to a *current* small open-weight model — **`Qwen/Qwen3-1.7B`**
(step down to `Qwen3-0.6B` for the cheapest/fastest deployments, up to `Qwen3-4B`/`Qwen3-8B` if
accuracy demands). Llama 3.2 (1B/3B, released 2024) is dated; the 2025–2026 small-model field
(Qwen3, Gemma 3, SmolLM3, Phi-3) beats it on most classification and instruction-following
benchmarks at equal size. §3.1 gives the full comparison and the rationale for Qwen3.

Databricks confirms Qwen support for both **Model Training** and **Model Serving** (docs updated
2026-09-22): the platform hosts Qwen3 variants, the custom-LLM serving path uses
`Qwen/Qwen2.5-VL-3B-Instruct` as its worked example, and provisioned throughput covers
fine-tuned/custom variants. Verify the exact variant's **region / preview status and license**
(most Qwen3 checkpoints are Apache-2.0) for your workspace before committing.

For pure fixed-schema classification, a small encoder (e.g. ModernBERT) is even cheaper, but we
stay on the decoder-LLM + logprobs path: it matches the request ("AI Runtime" fine-tuning of an
LLM) and generalizes across all three primitives with one recipe.

### 3.1 Open-weight model selection (2026)

| Model | Small sizes | License | Why / when to pick |
|-------|-------------|---------|--------------------|
| **Qwen3** ⭐ | 0.6B, 1.7B, 4B, 8B | Apache-2.0 (most ckpts) | **Default.** Widest small-size ladder, strong on structured/latency-sensitive classification, fine-tunes well, hosted on Databricks. Pick 0.6B/1.7B for the Jev-like speed/cost profile. |
| Gemma 3 | 1B, 4B | Gemma license | Strong instruction-following + multilingual at modest compute; check license terms. |
| SmolLM3 | 3B | Apache-2.0 (fully open: weights + training recipe) | Best when full transparency / reproducibility matters; good multilingual. |
| Phi-3-mini | 3.8B | MIT | Best raw accuracy in the SLM class (~69% MMLU); larger/slower than Qwen3-1.7B. |
| Llama 3.2 | 1B, 3B | Llama license | Widest runtime/quantization ecosystem, but **oldest** (2024) — trails the above at equal size. Keep only for ecosystem/compatibility reasons. |

**Recommendation:** start at **Qwen3-1.7B**; drop to **0.6B** if the eval holds (cheapest serving,
lowest latency — closest to Jev), or climb to **4B/8B** if accuracy on hard cases falls short.
Everything else in this spec is model-agnostic; only the `model=` string and the answer-token
tokenization check (§4, §12) change between families.

---

## 4. The Jev-like typed contract

The serving wrapper exposes one logical operation per primitive. All return a probability object,
never free text.

| Primitive | Input | Output | Backed by |
|-----------|-------|--------|-----------|
| **Choice** | prompt + option list `[A, B, C, ...]` | `{value: "B", probabilities: {A: .07, B: .88, C: .05}, confidence: .88}` | softmax over the logprobs of each option's answer token |
| **Score** | prompt + ordinal levels `1..K` | `{value: 4, probabilities: {1:.01,...,5:.02}, expected_score: 3.9}` | logprobs over level tokens (+ optional expected value) |
| **Noul** | prompt (a yes/no proposition) | `{probability: 0.73}` | calibrated `P(yes)` from the yes/no answer tokens |

**Key design rule:** every answer must be a **single token** from a small closed set, so one
forward pass yields the whole distribution. Use single-character / single-token labels
(`A`, `B`, `1`, `2`, `yes`→`Y`) and verify they tokenize to one id in the base tokenizer.

---

## 5. Phase 2 — Fine-tuning with AI Runtime (Mosaic AI Model Training)

> Databricks has consolidated fine-tuning under **AI Runtime / Databricks (Mosaic AI) Model
> Training**. It supports task types `INSTRUCTION_FINETUNE`, `CHAT_COMPLETION`, and
> `CONTINUED_PRETRAIN`, and supports the Qwen family (Qwen2.5 / Qwen3) alongside Gemma, Llama and
> others. We use **`INSTRUCTION_FINETUNE`** with (`prompt`, `response`) rows.

### 5.1 Data format (Phase 1 output)

Write JSONL to a UC Volume. `response` is exactly the single answer token, nothing else — this is
what makes the logprob read clean.

```json
{"prompt": "Classify the sentiment. Reply with one letter.\nOptions: A=negative B=neutral C=positive\nText: \"the wait was long but the food was worth it\"\nAnswer:", "response": " C"}
{"prompt": "Rate risk 1-5. Reply with one digit.\nTransaction: card-not-present, $4,300, new device, 3am\nAnswer:", "response": " 4"}
{"prompt": "Is this email phishing? Reply Y or N.\nSubject: \"Your account is locked, verify now\" ...\nAnswer:", "response": " Y"}
```

Guidelines:
- Keep the prompt template **identical** at train and inference time (the answer token position
  is what we read logprobs from).
- Balance classes; include hard/ambiguous cases so probabilities are meaningful, not saturated.
- **Synthetic / distilled data** (Jev's "synthetic only" analog): use a frontier teacher to
  label a large unlabeled corpus, or to *generate* labeled examples with rationale-then-label,
  then keep only the label. This is standard knowledge distillation and is the cheapest way to a
  large training set. Keep a **human-labeled held-out set** for calibration & evaluation (§8, §9).

### 5.2 Launch a training run

```python
# Databricks notebook — Mosaic AI Model Training (AI Runtime)
from databricks.model_training import foundation_model as fm

run = fm.create(
    model="Qwen/Qwen3-1.7B",                      # or Qwen3-0.6B (cheaper) / Qwen3-4B (more accurate)
    task_type="INSTRUCTION_FINETUNE",
    train_data_path="/Volumes/main/logprob/train/train.jsonl",
    eval_data_path="/Volumes/main/logprob/train/eval.jsonl",
    register_to="main.logprob.jev_like_qwen3",    # Unity Catalog model
    training_duration="3ep",                       # epochs; small models converge fast on narrow tasks
    learning_rate="5e-6",
    # data_prep_cluster_id / context_length as needed; confirm the exact Qwen variant string,
    # region/preview availability, and license in your workspace before running
)
run.wait()   # streams metrics; final model lands in UC registry
```

Notes:
- **LoRA/PEFT** keeps the run cheap (tens to low-hundreds of USD for a 1–2B model on a narrow
  task). Full fine-tune only if the task is far from the base distribution.
- Track loss/eval in the run; the registered model in UC gets full lineage.
- Because the label space is tiny, you need **far less data** than for open generation —
  thousands to low tens-of-thousands of examples often suffice.

---

## 6. Phase 4 — Deployment on Mosaic AI Model Serving

Serve the UC model on a **provisioned-throughput** endpoint (custom fine-tuned models are served
via provisioned throughput, billed in DBU/hour — see §11).

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput, ServedEntityInput,
)

w = WorkspaceClient()
w.serving_endpoints.create(
    name="jev-like-qwen3",
    config=EndpointCoreConfigInput(
        served_entities=[ServedEntityInput(
            entity_name="main.logprob.jev_like_qwen3",
            entity_version="1",
            # provisioned throughput band (tokens/sec) — size to your peak QPS
            min_provisioned_throughput=0,      # scale-to-zero for spiky/low volume
            max_provisioned_throughput=980,
        )],
    ),
)
```

### 6.1 Reading log-probabilities (the crux)

Databricks Foundation Model APIs expose `logprobs` and `top_logprobs` (0–20) — **on the
`completions` endpoint, not the chat endpoint.** So we query with the OpenAI-compatible
*completions* schema, set `max_tokens=1`, and read the distribution at the answer position.

```python
from openai import OpenAI
import os, math

client = OpenAI(
    base_url=f"{os.environ['DATABRICKS_HOST']}/serving-endpoints",
    api_key=os.environ["DATABRICKS_TOKEN"],
)

def choice(prompt: str, options: dict[str, str], T: float = 1.0):
    """options maps answer-token -> label, e.g. {' A':'negative',' B':'neutral',' C':'positive'}"""
    r = client.completions.create(
        model="jev-like-qwen3",
        prompt=prompt,
        max_tokens=1,
        temperature=0.0,
        logprobs=True,
        top_logprobs=20,          # covers the whole small answer space in one pass
    )
    top = r.choices[0].logprobs.top_logprobs[0]          # {token: logprob}
    # keep only our answer tokens, apply temperature T, renormalize
    logits = {tok: top.get(tok, -30.0) / T for tok in options}
    m = max(logits.values())
    exps = {tok: math.exp(v - m) for tok, v in logits.items()}
    Z = sum(exps.values())
    probs = {options[tok]: e / Z for tok, e in exps.items()}
    value = max(probs, key=probs.get)
    return {"value": value, "probabilities": probs, "confidence": probs[value]}
```

`Score` and `Noul` are the same read with different answer-token sets (digits, or `Y`/`N`); `Noul`
returns `P(Y)` directly and `Score` can also return an expected value `Σ level · p(level)`.

---

## 7. Phase 5 — The typed serving wrapper

Package the read + calibration + contract as an MLflow `pyfunc` model (or a small Databricks App /
AI Gateway route) so clients get `{value, probability, confidence}` and never touch raw logprobs.
The calibration temperature `T` (and any per-class Platt parameters) live in a UC table and are
loaded at serving time. This wrapper is the customer-facing "Jev" surface.

---

## 8. Phase 3 — Calibration (the "meaning" of the probability)

A raw softmax over logprobs is usually **over-confident**. To make the number trustworthy — Jev's
whole value proposition — fit a post-hoc calibrator on the **held-out labeled set**:

- **Temperature scaling** (Guo et al., 2017): fit a single scalar `T` minimizing NLL on held-out
  logits. Cheap, one parameter, does not change the argmax (so accuracy is unchanged), only the
  confidence. This is our default.
- **Platt / isotonic** per class if temperature scaling under-fits multi-class miscalibration.

Report calibration quality with **ECE** (expected calibration error), **Brier score**, and
**log-loss**, alongside accuracy/F1. Re-fit `T` on a schedule or when drift monitoring fires — a
Lakeflow job comparing recent prediction confidence vs. realized outcomes.

> **Two ways to read the distribution — and why calibration is mandatory.**
> (a) *Scoring mode:* score each candidate label under the model and normalize — preserves the
> model's native relative likelihoods. (b) *Constrained decode + logprobs* (what §6.1 does with a
> single forced token): fast and simple, **but hard token-masking changes the conditional
> distribution**, so the raw softmax is not directly comparable across inputs. Either way, the
> probability only *means* something after post-hoc calibration on held-out data — this is the
> step that turns "logprobs" into a Jev-like calibrated confidence. For high-stakes uses, wrap the
> calibrated scores in **conformal prediction** for distribution-free coverage guarantees.

---

## 9. Evaluation plan

| Dimension | Metric | Target / note |
|-----------|--------|---------------|
| Accuracy | top-1 accuracy / macro-F1 | ≥ frontier teacher on the narrow task |
| Calibration | ECE, Brier, log-loss | ECE materially lower after temperature scaling |
| Latency | p50 / p95 end-to-end | goal in Jev's 70–500 ms band for the 1B model |
| Throughput | calls/sec per PT band | drives the cost model (§11) |
| Cost | $ / 1M calls | vs. frontier baselines (§11) |

Run the eval as a Databricks job over the held-out set; log to MLflow so runs are comparable and
governed.

---

## 10. Demo walkthrough (what to show the customer)

1. **Problem:** a high-volume classification/scoring task (fraud triage, ticket routing, content
   safety, lead scoring) where they currently call a frontier chat API and parse text.
2. **Data prep:** distill labels with a teacher model into UC (5.1).
3. **Fine-tune:** one `fm.create(...)` call on Llama-3.2-1B; watch it register to UC (§5).
4. **Deploy:** provisioned-throughput endpoint (§6).
5. **The reveal:** same input to (a) the frontier chat API and (b) our endpoint — ours returns a
   **calibrated probability distribution in one token**, faster and cheaper.
6. **Calibration chart:** reliability diagram before/after temperature scaling (§8).
7. **Cost slide:** the §11 table for their volume, with the break-even.

---

## 11. Cost comparison

> **All figures illustrative, gathered 2026-09-23; verify before quoting.** DBU→USD conversion is
> region/contract dependent. Model Serving DBU list price used here: **$0.070/DBU** (placeholder —
> substitute the customer's actual rate). Frontier API prices below are Sept-2026 public list
> prices per **1M tokens (input / output)**.

### 11.1 Workload assumptions

- Task: single-label classification / scoring.
- **Input ≈ 150 tokens/call**, **output = 1 token** (the class) + logprobs.
- Frontier "chat used as classifier" pays for input every call; our model emits exactly 1 output
  token (a chat model told to "explain then answer" would emit far more — a hidden API cost).

### 11.2 Per-call cost of the frontier baselines

| Model (Sept 2026 list) | $/1M in | $/1M out | $/call (150 in + 1 out) |
|---|---|---|---|
| GPT-4o-mini | 0.15 | 0.60 | $0.0000231 |
| GPT-5.6 "Luna" | 0.20 | 1.20 | $0.0000312 |
| Claude Haiku 4.5 | 1.00 | 5.00 | $0.000155 |
| GPT-5 (chat) | 1.25 | 10.00 | $0.0001975 |

### 11.3 OSS model on Databricks (Qwen3-1.7B, provisioned throughput)

- Serving rate: use the small-model PT band as a proxy — Databricks lists **85.7 DBU/hr**
  (Llama-3.2-1B) to **92.9 DBU/hr** (Llama-3.2-3B); a 1.7B model sits between. Take **~90 DBU/hr**
  → ≈ **$6.30/hr** at $0.070/DBU → ≈ **$4,600/month** running 24×7 (730 hr). *(Confirm the exact
  Qwen3 PT rate for your region — Qwen-specific DBU rates were not verified here.)*
- A 1.7B model is tiny; **one PT band** sustains a high token rate. Illustratively (verify with a
  load test) one band handles on the order of **~100M calls/month** of 150-token classification
  calls. Below that, **scale-to-zero** cuts idle cost; above it, add bands ~linearly.
- **One-time fine-tune** (LoRA, 1–2B, narrow task): ~**$50–$300** in Model Training DBUs.
  Amortized to ~$0 per call at any real volume.

### 11.4 Monthly cost by volume ($/month; OSS = flat provisioned capacity)

| Calls/month | GPT-4o-mini | Luna | Claude Haiku | GPT-5 chat | **OSS Qwen3-1.7B (Databricks)** |
|---|---|---|---|---|---|
| 10M | $231 | $312 | $1,550 | $1,975 | ~$1,600 (scaled down) |
| 100M | $2,310 | $3,120 | $15,500 | $19,750 | **~$4,600 (1 band 24×7)** |
| 1B | $23,100 | $31,200 | $155,000 | $197,500 | **~$46,000 (~10 bands)** |

### 11.5 Break-even & interpretation

Break-even against a flat ~$4,600/mo OSS band (`N* = fixed_cost / per_call_api_cost`):

- vs **GPT-5 chat**: ≈ **23M calls/mo** — above this, OSS is cheaper.
- vs **Claude Haiku**: ≈ **30M calls/mo**.
- vs **GPT-4o-mini / Luna** (the cheapest tiers): only ≈ **200M calls/mo**.

**Honest read of the numbers:**
- The dramatic "40–400× cheaper" gap Jev advertises is realistic **against premium chat tiers**
  (GPT-5, Haiku) at high volume — that is where OSS-on-Databricks shines.
- Against the **cheapest** frontier tiers, the API wins on pure $/call until you reach very high,
  steady volume. The OSS case there rests on the **non-price** advantages below.
- The comparison also **understates** the API cost whenever a chat model is prompted to reason in
  text before answering (every reasoning token is billed; our model emits one).

**Non-price advantages (often the real decision drivers):**
- **Data governance / residency:** inputs never leave the customer's Databricks + UC boundary.
- **Latency & determinism:** small model, one token, no external network hop → Jev-like 70–500 ms;
  `temperature=0` gives reproducible probabilities.
- **No rate limits / no vendor lock-in / no silent model swaps** under you.
- **Calibrated by construction:** the probability is fit to *their* outcomes, not a generic model's.

---

## 12. OSS landscape: building blocks comparable to Jev

Jev is a single closed product, but its *capability* — typed value + calibrated probability — is
assembled from open, composable pieces. There is no single open-weight "Jev"; you build the
equivalent from three layers, all of which run on/with Databricks:

**Layer 1 — the base model (open weights).** Qwen3 (0.6B–8B), Gemma 3, SmolLM3, Phi-3, Llama 3.2.
See §3.1. This spec fine-tunes one of these; that's the "learned probabilities" part.

**Layer 2 — structural guarantees (constrained decoding / structured generation).** These force
the output into a closed schema — a single label token, a JSON object, a grammar — so there is
never a parse failure, matching Jev's "no type errors" promise:
- **Outlines** (dottxt-ai) — Python-first; JSON Schema / Pydantic / regex / CFG; backends for
  transformers, vLLM, llama.cpp. Best for product integration.
- **XGrammar / XGrammar-2** (mlc-ai) — grammar/automaton engine co-designed with inference engines
  for near-zero serving overhead; supports richer (pushdown) grammars.
- **LLGuidance** (guidance-ai) — fast token-masking / grammar primitives.
- For our flat closed-choice case, a **trie/token-mask over the label set** is the simplest and
  fastest option — and is effectively what §6.1 does with `top_logprobs` over single tokens.

**Layer 3 — calibrated confidence (the part that makes the number trustworthy).**
- Post-hoc calibration: **temperature scaling** (default), Platt, isotonic (§8).
- **Conformal prediction** for coverage guarantees on high-stakes decisions.
- Metrics: ECE, Brier, log-loss, reliability diagrams.

**Key nuance from the literature:** structural correctness ≠ semantic correctness, and hard
masking distorts the native distribution (§8). Small models can obey the grammar yet be wrong, and
their raw probabilities are over-confident — so **Layer 3 is not optional**. Jev's RLCD folds
calibration into training; our open stack achieves the same end by fine-tuning (Layer 1) +
constraining (Layer 2) + calibrating (Layer 3), each independently swappable and governed in UC.

### 12.1 Runnable example — constrained decoding + calibrated probabilities

These run locally (e.g. in a Databricks GPU notebook against the fine-tuned Qwen3 checkpoint) and
are the **dev-loop mirror** of the served endpoint in §6. APIs verified against Outlines v1 and
XGrammar (2026).

```bash
pip install "outlines>=1.0" xgrammar transformers torch pydantic
```

**A — Outlines: guaranteed-valid label (Layer 2, structural guarantee).**
The Generator can only emit one of the allowed labels — no parsing, no repair.

```python
from typing import Literal
from outlines import Generator, models

model = models.transformers("Qwen/Qwen3-1.7B", device="cuda")   # your fine-tuned ckpt path also works
Label = Literal["negative", "neutral", "positive"]              # closed answer space

classify = Generator(model, Label)
label = classify(
    "Classify the sentiment.\nText: 'the wait was long but the food was worth it'\nAnswer:",
    max_new_tokens=8, temperature=0.0,
).strip()
print(label)   # -> one of the three labels, guaranteed
```

**B — XGrammar: schema-constrained structured output (Layer 2, richer grammars).**
XGrammar's HF `LogitsProcessor` enforces a grammar/JSON-schema at decode time with near-zero
overhead — use it when the typed value is an object, not a single label.

```python
import torch, xgrammar as xgr
from transformers import AutoTokenizer, AutoModelForCausalLM

name = "Qwen/Qwen3-1.7B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda").eval()

ti = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=model.config.vocab_size)
compiler = xgr.GrammarCompiler(ti)
# a "Score" primitive as a grammar: one digit 1-5
compiled = compiler.compile_grammar(r'root ::= "1" | "2" | "3" | "4" | "5"')
lp = xgr.contrib.hf.LogitsProcessor(compiled)

prompt = "Rate risk 1-5.\nTransaction: card-not-present, $4,300, new device, 3am\nAnswer:"
ins = tok(prompt, return_tensors="pt").to("cuda")
out = model.generate(**ins, max_new_tokens=2, do_sample=False, logits_processor=[lp])
print(tok.decode(out[0][ins.input_ids.shape[1]:], skip_special_tokens=True))   # a valid digit
```

**C — The Jev-like part: a *calibrated probability distribution*, not just a sample.**
Constrained sampling (A/B) guarantees the *shape*; it does not give a trustworthy *number*. For a
closed label set, do **one forward pass**, read the logits over the label tokens, and apply the
temperature `T` fit in §8. This is the local equivalent of the §6.1 serving read and produces the
`{value, probabilities, confidence}` contract.

```python
import torch, torch.nn.functional as F

LABELS = {"negative": " negative", "neutral": " neutral", "positive": " positive"}
# first token id of each label (verify single-token or use first-token scoring; see §12 tokenization note)
LABEL_IDS = {k: tok(v, add_special_tokens=False).input_ids[0] for k, v in LABELS.items()}

def jev_choice(prompt: str, T: float = 1.0):
    ins = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**ins).logits[0, -1, :]         # next-token logits at the answer position
    ids = torch.tensor(list(LABEL_IDS.values()), device=logits.device)
    probs = F.softmax(logits[ids] / T, dim=-1)          # temperature-scaled over the label set only
    dist = {lab: probs[i].item() for i, lab in enumerate(LABEL_IDS)}
    value = max(dist, key=dist.get)
    return {"value": value, "probabilities": dist, "confidence": dist[value]}

print(jev_choice("Classify the sentiment.\nText: 'never coming back'\nAnswer:", T=1.7))
# -> {'value': 'negative', 'probabilities': {...}, 'confidence': 0.94}
```

`T` comes from the held-out calibration fit (§8); `T > 1` softens the over-confident raw softmax.
Swap the label set for digits (`Score`) or `Y`/`N` (`Noul`) to cover the other two primitives.

---

## 13. Risks, caveats & open questions

- **Single-token labels must tokenize to one id** in the base tokenizer — verify per model; pick
  labels accordingly (leading-space variants like `" A"` often differ from `"A"`).
- **`logprobs` is on the completions endpoint, not chat** — the wrapper must use the completions
  schema. Confirm this still holds for the chosen serving stack at build time.
- **Provisioned-throughput sizing** and the ~100M-calls/band figure are **illustrative** —
  run a real load test to fix the cost model before quoting.
- **$/DBU and API prices move** — all §11 numbers are 2026-09-23 placeholders.
- **Calibration drift:** distribution shift degrades `T`; schedule re-calibration + monitoring.
- **RLCD is not reproduced** — we approximate its intent (outcome-calibrated probabilities) with
  distillation + temperature scaling, which is sufficient for the demo but is not Jev's method.
- **Accuracy ceiling:** a 1B model may trail the teacher on hard cases; step up to 3B/8B or
  improve the distilled dataset if eval falls short.

---

## 14. References

- **Jev (AI model)** — Wikipedia: <https://en.wikipedia.org/wiki/Jev_(AI_model)> (read 2026-09-23).
- **(unavailable)** ChatGPT share link from the request:
  `https://chatgpt.com/s/t_6ab0fbe0d5b481918375ae5ee1017d23` — requires auth, not fetched.
- **Databricks Model Training / Foundation Model Fine-tuning** (task types, supported models incl.
  Qwen family): Databricks docs, *large-language-models/foundation-model-training* and
  *model-serving/foundation-model-overview* / *serve-custom-llms* (Qwen support confirmed
  2026-09-22).
- **Open-weight small models (2026)**: Qwen3 technical report (arXiv 2505.09388); Gemma 3
  (DeepMind); SmolLM3 (Hugging Face); Phi-3 (arXiv 2404.14219); Llama 3.2 model card.
- **Structured generation / constrained decoding**: Outlines (dottxt-ai), XGrammar / XGrammar-2
  (mlc-ai), LLGuidance (guidance-ai); JSONSchemaBench.
- **Foundation Model APIs — logprobs / top_logprobs**: Databricks docs,
  *machine-learning/foundation-model-apis/api-reference* and *model-serving-query* API.
- **Mosaic AI Model Serving pricing (DBU rates)**: Databricks pricing / Foundation Model APIs docs.
- **Frontier API list prices (Sept 2026)**: OpenAI and Anthropic pricing pages.
- **Calibration**: Guo, Pleiss, Sun, Weinberger, *On Calibration of Modern Neural Networks*,
  ICML 2017 (temperature scaling, ECE).
