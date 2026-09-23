# Literature Survey: Calibrated, Typed, Log-Prob LLMs

**Companion to [`SPECS.md`](./SPECS.md).** This is the comprehensive related-work survey behind the
spec for building an open-source "log-prob LLM" on Databricks — a small fine-tuned model that
returns a *calibrated probability* over a closed answer space instead of prose. It covers the
research the design rests on: confidence calibration, uncertainty quantification, hallucination
detection, selective prediction / abstention, conformal prediction, constrained decoding, and
knowledge distillation into small models.

**Last updated:** 2026-09-23.

---

## How this survey was verified (read this first)

Every arXiv entry below was confirmed by **fetching its arXiv abstract page** and checking that the
title, authors, and ID match — no IDs were cited from memory or from an LLM's unverified guess.

- The fetch mechanism was **control-tested**: a deliberately nonexistent ID (`arXiv:2609.99999`)
  returns HTTP 404, and a plausible-but-fake title does not materialize — so a successful fetch is
  real evidence the paper exists.
- Verification legend:
  - **✓✓** — independently re-fetched and confirmed by the spec author.
  - **✓** — confirmed by a research agent's direct arXiv fetch (mechanism control-tested above);
    a representative sample of these was additionally re-checked by the author and all passed.
- **Scope & honesty.** This is a *curated* comprehensive survey, not an exhaustive dump. The
  research pass surfaced ~120 candidates; clearly off-topic items (e.g. LLM-voting/collective-choice
  papers) and the long tail of niche 2025–2026 preprints were dropped to keep signal high. Dates
  past the author's 2026-01 knowledge cutoff (several 2026 preprints) rely on the fetch check, not
  prior knowledge. Peer-reviewed venues are noted where known; everything else is an arXiv preprint.

**Map to the spec:** §1→SPECS §8 (calibration); §2–§3→SPECS §8 note & §12 (why raw softmax needs
calibration); §4→SPECS §8–§9 (abstention/routing); §5→SPECS §8 (conformal wrapper); §6→SPECS §12
(structural layer); §7→SPECS §3.1 & §5 (base model + distillation).

---

## 1. Confidence calibration

**Foundations.**
- **Guo, Pleiss, Sun & Weinberger (2017)**, *On Calibration of Modern Neural Networks*. ICML. arXiv:1706.04599 ✓✓ — Modern nets are miscalibrated; **temperature scaling** (one scalar) largely fixes it. The method used in SPECS §8.
- **Nixon et al. (2019)**, *Measuring Calibration in Deep Learning*. arXiv:1904.01685 ✓ — Flaws in ECE and better ways to measure calibration; informs the metric choice in SPECS §9.
- **Minderer et al. (2021)**, *Revisiting the Calibration of Modern Neural Networks*. NeurIPS. arXiv:2106.07998 ✓ — Newer architectures calibrate better than the 2017 picture suggested; relevant to picking a modern base model.
- **Desai & Durrett (2020)**, *Calibration of Pre-trained Transformers*. arXiv:2003.07892 ✓✓ — BERT/RoBERTa are well-calibrated in-domain; temperature scaling / label smoothing help under shift. Justifies applying §8 to a transformer classifier.
- **Zhao et al. (2021)**, *Calibrate Before Use: Improving Few-Shot Performance of Language Models*. ICML. arXiv:2102.09690 ✓ — Contextual calibration removes prompt/label bias in few-shot LMs.
- **Manokhin & Grønhaug (2026)**, *Classifier Calibration at Scale: An Empirical Study of Model-Agnostic Post-Hoc Methods*. arXiv:2601.19944 ✓✓ — Benchmarks post-hoc calibrators across 21 classifiers; Venn-Abers often beats Platt/isotonic — a concrete alternative if temperature scaling underfits (SPECS §8).

**LLM-specific calibration.**
- **Jiang et al. (2021)**, *How Can We Know When Language Models Know? …*. TACL. arXiv:2012.00955 ✓ — Early study of whether LM probabilities track QA correctness; identifies calibration gaps.
- **Kadavath et al. (2022)**, *Language Models (Mostly) Know What They Know*. arXiv:2207.05221 ✓✓ — Larger LMs are well-calibrated on multiple-choice / true-false **when the answer space is constrained and well-formatted** — exactly the closed-token setup (SPECS §4).
- **Tian et al. (2023)**, *Just Ask for Calibration …*. arXiv:2305.14975 ✓✓ — On RLHF-tuned models, *verbalized* confidence can beat conditional probabilities (~50% lower ECE). The alternative flagged in SPECS §8.
- **Xiong et al. (2023)**, *Can LLMs Express Their Uncertainty? …*. ICLR 2024. arXiv:2306.13063 ✓✓ — Black-box benchmark of prompt/sample/aggregate confidence; consistency-based methods beat pure verbalization.
- **He et al. (2023)**, *Investigating Uncertainty Calibration of Aligned Language Models under the Multiple-Choice Setting*. arXiv:2310.11732 ✓ — Aligned LLMs are overconfident on MC; a post-hoc method separates two uncertainty sources.
- **Zhang et al. (2024)**, *Calibrating the Confidence of LLMs by Eliciting Fidelity*. EMNLP. arXiv:2404.02655 ✓ — Decomposes confidence into uncertainty + fidelity across RLHF models.
- **Yu et al. (2026)**, *ChainUQ: Reasoning Consistency-Aware Uncertainty Quantification for LLMs*. arXiv:2609.26060 ✓✓ — Refines confidence via reasoning-chain consistency; reports ~45% relative ECE reduction.

**Survey.**
- **Geng et al. (2023)**, *A Survey of Confidence Estimation and Calibration in Large Language Models*. arXiv:2311.08298 ✓✓ — The reference survey for this section.

---

## 2. Verbalized, sampling-based & semantic uncertainty

**Verbalized / elicited confidence.**
- **Lin, Hilton & Evans (2022)**, *Teaching Models to Express Their Uncertainty in Words*. arXiv:2205.14334 ✓ — GPT-3 can emit calibrated verbalized probabilities; foundational for "just ask."
- **Yang et al. (2023)**, *Improving the Reliability of LLMs by Leveraging Uncertainty-Aware In-Context Learning*. arXiv:2310.04782 ✓ — Uses logit-reflected uncertainty in-context so the model flags uncertain cases.

**Sampling / consistency & semantic entropy.**
- **Wang et al. (2022)**, *Self-Consistency Improves Chain of Thought Reasoning …*. ICLR 2023. arXiv:2203.11171 ✓ — Sample-and-vote; the basis of consistency-based confidence.
- **Kuhn, Gal & Farquhar (2023)**, *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in NLG*. ICLR 2023. arXiv:2302.09664 ✓ — **Semantic entropy** clusters meaning-equivalent samples; a stronger UQ signal than token entropy for free-form output.
- **Lin, Trivedi & Sun (2023)**, *Generating with Confidence: Uncertainty Quantification for Black-box LLMs*. arXiv:2305.19187 ✓ — Confidence from sample disagreement without logits.
- **Xie et al. (2024)**, *Calibrating Reasoning in Language Models with Internal Consistency*. arXiv:2405.18711 ✓ — Intermediate-layer agreement as a calibration signal; detects reasoning-vs-answer mismatch.
- **Hou et al. (2024)**, *Decomposing Uncertainty for LLMs through Input Clarification Ensembling*. ICML. arXiv:2311.08718 ✓ — Separates aleatoric vs epistemic uncertainty via clarification ensembles.

**Surveys.**
- **Hu et al. (2023)**, *Uncertainty in Natural Language Processing: Sources, Quantification, and Applications*. arXiv:2306.04459 ✓ — Taxonomy of input/system/output uncertainty in NLP.
- **Huang et al. (2024)**, *A Survey of Uncertainty Estimation in LLMs: Theory Meets Practice*. arXiv:2410.15326 ✓ — Bayesian / information-theoretic framing with practical applications.
- **Shorinwa et al. (2024)**, *A Survey on Uncertainty Quantification of LLMs: Taxonomy, Open Research Challenges, and Future Directions*. arXiv:2412.05563 ✓✓ — Broad, current UQ taxonomy.

---

## 3. Hallucination & factuality detection

- **Manakul, Liusie & Gales (2023)**, *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection …*. EMNLP. arXiv:2303.08896 ✓ — Cross-sample inconsistency signals hallucination; no model access needed.
- **Chen & Mueller (2023)**, *Quantifying Uncertainty in Answers from any Language Model …* (BSDetector). arXiv:2308.16175 ✓ — Black-box confidence from semantic similarity across samples.
- **Duan et al. (2023)**, *Shifting Attention to Relevance (SAR) …*. ACL 2024. arXiv:2307.01379 ✓ — Relevance-weighted token uncertainty for free-form QA.
- **Fadeeva et al. (2023)**, *LM-Polygraph: Uncertainty Estimation for Language Models*. EMNLP. arXiv:2311.07383 ✓ — Open framework implementing many token/sequence UE methods.
- **Fadeeva et al. (2024)**, *Fact-Checking the Output of LLMs via Token-Level Uncertainty Quantification*. arXiv:2403.04696 ✓ — Claim-Conditioned Probability isolates factual-claim uncertainty.
- **Vashurin et al. (2024)**, *Benchmarking Uncertainty Quantification Methods for LLMs with LM-Polygraph*. TACL 2025. arXiv:2406.15627 ✓ — Standardized UQ benchmark across 11 tasks.
- **Karbasi et al. (2025)**, *Impossibility of Automated Hallucination Detection in Large Language Models*. arXiv:2504.17004 ✓ — Theory: detection from positive examples alone is impossible; RLHF-style negatives are needed. A caution for any confidence-only guardrail.
- **Farquhar et al. (2024)**, *Detecting hallucinations in large language models using semantic entropy*. Nature 630:625–630 (no arXiv) ✓ — Semantic entropy for hallucination detection, at Nature-review bar.

**Surveys.**
- **Huang et al. (2023)**, *A Survey on Hallucination in LLMs: Principles, Taxonomy, Challenges, and Open Questions*. arXiv:2311.05232 ✓ — The widely-cited hallucination survey.
- **Wang et al. (2024)**, *Factuality of Large Language Models: A Survey*. arXiv:2402.02420 ✓ — Causes, evaluation, and mitigation of factual errors.

---

## 4. Selective prediction, abstention & learning to defer

- **El-Yaniv & Wiener (2010)**, *On the Foundations of Noise-free Selective Classification*. JMLR (no arXiv) — Canonical risk–coverage theory for predict-or-abstain.
- **Geifman & El-Yaniv (2017)**, *Selective Classification for Deep Neural Networks*. arXiv:1705.08500 ✓✓ — Confidence-thresholded abstention at a guaranteed risk level; the framework for routing low-confidence outputs to a human (SPECS §8–§9).
- **Geifman & El-Yaniv (2019)**, *SelectiveNet: A Deep Neural Network with an Integrated Reject Option*. ICML. arXiv:1901.09192 ✓ — Learns classification + rejection jointly.
- **Kamath, Jia & Liang (2020)**, *Selective Question Answering under Domain Shift*. ACL. arXiv:2006.09462 ✓✓ — A trained calibrator decides when to abstain under shift — the pattern behind drift-aware routing.
- **Ren et al. (2023)**, *Out-of-Distribution Detection and Selective Generation for Conditional Language Models*. ICLR 2023. arXiv:2209.15558 ✓✓ — Lightweight OOD detector for CLMs to gate generation.
- **Ovadia et al. (2019)**, *Can You Trust Your Model's Uncertainty? …*. NeurIPS. arXiv:1906.02530 ✓ — Uncertainty degrades under dataset shift; ensembles are more robust than calibration alone. Motivates monitoring + recalibration (SPECS §8).

**Survey.**
- **Wen et al. (2024)**, *Know Your Limits: A Survey of Abstention in Large Language Models*. TACL 2024. arXiv:2407.18418 ✓✓ — Comprehensive treatment of when and how LLMs should abstain.

---

## 5. Conformal prediction & distribution-free UQ

**Foundations.**
- **Vovk, Gammerman & Shafer (2005)**, *Algorithmic Learning in a Random World*. Springer (no arXiv) — The founding monograph.
- **Shafer & Vovk (2007)**, *A Tutorial on Conformal Prediction*. JMLR. arXiv:0706.3188 ✓ — Accessible entry point.
- **Tibshirani, Foygel Barber, Candès & Ramdas (2019)**, *Conformal Prediction Under Covariate Shift*. NeurIPS. arXiv:1904.06019 ✓ — Weighted conformal for train/test mismatch — directly relevant to production drift.
- **Angelopoulos & Bates (2021)**, *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv:2107.07511 ✓✓ — The practical tutorial; the wrapper referenced in SPECS §8.
- **Angelopoulos et al. (2022)**, *Conformal Risk Control*. arXiv:2208.02814 ✓ — Extends conformal to control any monotone loss, not just miscoverage.
- **Barber (2024)**, *Theoretical Foundations of Conformal Prediction*. Cambridge Univ. Press. arXiv:2411.11824 ✓ — Unified modern treatment.

**Conformal for LLMs / NLP.**
- **Kumar et al. (2023)**, *Conformal Prediction with Large Language Models for Multi-Choice Question Answering*. ICML workshop. arXiv:2305.18404 ✓ — Prediction sets for MCQA; studies exchangeability violations.
- **Ravfogel, Goldberg & Goldberger (2023)**, *Conformal Nucleus Sampling*. ACL 2023. arXiv:2305.02633 ✓ — Calibrates top-p sampling; exposes overconfidence.
- **Quach et al. (2023)**, *Conformal Language Modeling*. ICLR 2024. arXiv:2306.10193 ✓✓ — Calibrated stopping/rejection with coverage guarantees for generative LMs.
- **Ren et al. (2023)**, *Robots That Ask For Help: Uncertainty Alignment for LLM Planners* (KnowNo). CoRL. arXiv:2307.01928 ✓ — Conformal sets trigger a reliable "ask for help" — a concrete abstention application.
- **Mohri & Hashimoto (2024)**, *Language Models with Conformal Factuality Guarantees*. ICML. arXiv:2402.10978 ✓ — Back-off to achieve target correctness on closed-book QA.
- **Su et al. (2024)**, *API Is Enough: Conformal Prediction for LLMs Without Logit-Access*. arXiv:2403.01216 ✓ — Conformal UQ using only API outputs — useful when logprobs are unavailable.
- **Wang et al. (2024)**, *ConU: Conformal Uncertainty in LLMs with Correctness Coverage Guarantees*. EMNLP. arXiv:2407.00499 ✓ — Self-consistency-based conformal for black-box LLMs.

**Surveys.**
- **Campos et al. (2024)**, *Conformal Prediction for Natural Language Processing: A Survey*. arXiv:2405.01976 ✓✓ — The NLP-focused conformal survey.
- **Zhou et al. (2024)**, *Conformal Prediction: A Data Perspective*. arXiv:2410.06494 ✓ — Data-centric view across structured/unstructured/dynamic data.

---

## 6. Constrained & structured decoding (the structural-guarantee layer)

- **Scholak, Schucher & Bahdanau (2021)**, *PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding …*. EMNLP. arXiv:2109.05093 ✓ — Incremental parsing rejects invalid tokens (SQL); early constrained decoding.
- **Beurer-Kellner, Fischer & Vechev (2023)**, *Prompting Is Programming: A Query Language for LLMs* (LMQL). arXiv:2212.06094 ✓ — Declarative constraints over generation.
- **Geng et al. (2023)**, *Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning*. arXiv:2305.13971 ✓ — Grammar masks eliminate syntax errors with no fine-tuning.
- **Willard & Louf (2023)**, *Efficient Guided Generation for Large Language Models*. arXiv:2307.09702 ✓✓ — FSM-indexing method behind **Outlines** (SPECS §12.1-A).
- **Ugare et al. (2024)**, *SynCode: LLM Generation with Grammar Augmentation*. arXiv:2403.01632 ✓ — DFA-based enforcement; zero JSON syntax errors.
- **Beurer-Kellner et al. (2024)**, *Guiding LLMs The Right Way: Fast, Non-Invasive Constrained Generation* (DOMINO). arXiv:2403.06988 ✓ — Subword-aligned constraints at near-zero overhead.
- **Park et al. (2024)**, *Grammar-Aligned Decoding*. arXiv:2405.21047 ✓ — Grammar-constrained sampling that **preserves the model's distribution** — directly addresses the masking-distortion caveat in SPECS §8/§12.
- **Dong et al. (2024)**, *XGrammar: Flexible and Efficient Structured Generation Engine for LLMs*. arXiv:2411.15100 ✓✓ — CFG decoding with up to ~100× speedup — **XGrammar** (SPECS §12.1-B).
- **Geng et al. (2025)**, *JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models*. arXiv:2501.10868 ✓ — Benchmarks constrained-decoding engines on ~9.5k real JSON schemas.

**Semantic vs. structural correctness.**
- **Chavan (2026)**, *Constrained Decoding Eliminates Structural Failures in Small LLMs but Reveals a Scale-Dependent Semantic Gap*. arXiv:2609.23742 ✓✓ — Constrained decoding fixes schema validity but **not** content correctness in small models; the empirical basis for "Layer 2 guarantees shape, not truth" (SPECS §12).

---

## 7. Knowledge distillation & small open-weight models

**Distillation** (how to build the training set — SPECS §5.1).
- **Hinton, Vinyals & Dean (2015)**, *Distilling the Knowledge in a Neural Network*. arXiv:1503.02531 ✓ — Soft-target distillation; the founding idea.
- **Sanh et al. (2019)**, *DistilBERT …*. arXiv:1910.01108 ✓ — ~40% smaller, ~97% of BERT — proof small distilled classifiers work in production.
- **Hsieh et al. (2023)**, *Distilling Step-by-Step! …*. ACL Findings. arXiv:2305.02301 ✓ — Distills teacher rationales so a 770M model beats far larger ones with less data.

**Small open-weight models** (base-model choices — SPECS §3.1).
- **Mesnard et al. / Gemma Team (2024)**, *Gemma: Open Models Based on Gemini Research and Technology*. arXiv:2403.08295 ✓
- **Abdin et al. (2024)**, *Phi-3 Technical Report …*. arXiv:2404.14219 ✓✓
- **Grattafiori et al. (2024)**, *The Llama 3 Herd of Models*. arXiv:2407.21783 ✓
- **Riviere et al. / Gemma Team (2024)**, *Gemma 2: Improving Open Language Models at a Practical Size*. arXiv:2408.00118 ✓ — Uses distillation for its smaller variants.
- **Warner et al. (2024)**, *ModernBERT: Smarter, Better, Faster, Longer …*. arXiv:2412.13663 ✓ — The efficient encoder alternative noted in SPECS §3.
- **Ben Allal et al. (2025)**, *SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model*. arXiv:2502.02737 ✓
- **Kamath et al. / Gemma Team (2025)**, *Gemma 3 Technical Report*. arXiv:2503.19786 ✓
- **Yang et al. / Qwen Team (2025)**, *Qwen3 Technical Report*. arXiv:2505.09388 ✓✓ — The 0.6B–235B family; small variants are the SPECS §3.1 default.
- **El Abd et al. / Gemma Team (2026)**, *Gemma 4 Technical Report*. arXiv:2607.02770 ✓✓ — Latest small multimodal open-weight family.

---

## Deliberately out of scope

To keep the survey focused, the research pass also surfaced but **excluded**: LLM-voting /
collective-choice papers; several domain-specific hallucination benchmarks (medical, multilingual
news, multimodal image/video); self-consistency *cost-optimization* variants; and a long tail of
niche 2025–2026 conformal preprints (pipeline-aware, routing/cascade, temporal-logic,
compression-under-uncertainty). These are real and fetch-verified but tangential to a calibrated
closed-answer classifier. Ask if you want any of these threads expanded into their own section.
