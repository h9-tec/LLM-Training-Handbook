# LLM Training Handbook

**A technical reference on how large language models are trained: the mechanisms, the economics, and the operational record of large-scale training runs.**

Most training tutorials stop at the loss function. This handbook takes the opposite approach: every chapter is anchored in **documented training runs**, including the OPT-175B logbook, the Llama 3.1 405B reliability data, DeepSeek-V3's FP8 recipe, Kimi K2's run without loss spikes, OLMo 2's stability investigation, SmolLM3's restart after one trillion tokens, the Tulu 3 and DeepSeek-R1 post-training pipelines, and the Arabic adaptation literature (Jais, AceGPT, ALLaM, Kuwain). A reader who finishes it should be able to read a training log or a technical report critically, and to recognize a divergence, a throughput regression, or a contaminated evaluation from its signature.

It is the training-side companion to the [LLM-Inference-Handbook](https://github.com/h9-tec/LLM-Inference-Handbook) and the [LLM-Math-Handbook](https://github.com/h9-tec/LLM-Math-Handbook). The inference handbook covers how a model is served economically. This one covers how the model came to exist and what failed along the way.

**Audience.** ML engineers, infrastructure engineers, and technical leads who train, fine-tune, or adapt LLMs, or who need to evaluate training reports. Candidates preparing for training and infrastructure interviews will find the standard questions covered with real numbers.

**Prerequisites.** Nothing to install. Familiarity with basic machine learning and the transformer architecture is assumed. The Arabic-specific sections assume no prior knowledge of Arabic NLP.

**Conventions.** Figures are quoted from the cited reports. Where a figure is derived here, the text says so and shows the arithmetic. Each chapter ends with a linked reference list, and cross-references use chapter (Ch.) and section (Sec.) numbers.

---

## Contents

**Part I: The Map**

1. [What "Training" Means in 2026: The Full Pipeline](#1-what-training-means-in-2026-the-full-pipeline)
2. [The Economics of Training](#2-the-economics-of-training)

**Part II: Pretraining**

3. [Data Engineering: The FineWeb Methodology](#3-data-engineering-the-fineweb-methodology)
4. [Tokenizers and the Arabic Tokenization Tax](#4-tokenizers-and-the-arabic-tokenization-tax)
5. [Scaling Laws, Budgets, and Ablation Methodology](#5-scaling-laws-budgets-and-ablation-methodology)
6. [Architecture Choices as Training Decisions](#6-architecture-choices-as-training-decisions)
7. [Optimizers, Schedules, and Hyperparameters](#7-optimizers-schedules-and-hyperparameters)
8. [Training Stability: Loss-Spike Mechanisms and Mitigations](#8-training-stability-loss-spike-mechanisms-and-mitigations)
9. [Numerical Precision: FP16, BF16, and FP8](#9-numerical-precision-fp16-bf16-and-fp8)
10. [Distributed Training: Parallelism Strategies and Measured MFU](#10-distributed-training-parallelism-strategies-and-measured-mfu)
11. [Hardware Reliability at Scale](#11-hardware-reliability-at-scale)

**Part III: Mid-Training and Adaptation**

12. [Mid-Training: Annealing, Curricula, and Long Context](#12-mid-training-annealing-curricula-and-long-context)
13. [Continued Pretraining and New-Language Adaptation: The Arabic Case](#13-continued-pretraining-and-new-language-adaptation-the-arabic-case)

**Part IV: Post-Training**

14. [Supervised Fine-Tuning](#14-supervised-fine-tuning)
15. [Preference Optimization and the GPT-4o Sycophancy Incident](#15-preference-optimization-and-the-gpt-4o-sycophancy-incident)
16. [RL for Reasoning: GRPO, RLVR, and the R1 Pipeline](#16-rl-for-reasoning-grpo-rlvr-and-the-r1-pipeline)

**Part V: The Practitioner's Track**

17. [Fine-Tuning Without a Cluster: LoRA, Full Fine-Tuning, and a Decision Procedure](#17-fine-tuning-without-a-cluster-lora-full-fine-tuning-and-a-decision-procedure)
18. [Evaluation During Training](#18-evaluation-during-training)
19. [Incident Index and Pre-Flight Checklists](#19-incident-index-and-pre-flight-checklists)
20. [Sources and Further Reading](#20-sources-and-further-reading)

---

# Part I: The Map

## 1. What "Training" Means in 2026: The Full Pipeline

When a laboratory reports that it "trained a model," the statement covers a pipeline of four to six distinct phases, each with its own data, hyperparameters, failure modes, and often its own team. Treating these phases as one activity is the most common source of confusion when reading technical reports.

```
raw web + curated sources
        |
        v
[ Data engineering ]   filtering, deduplication, classification, mixing      Ch. 3
        |
        v
[ Pretraining ]        next-token prediction on trillions of tokens          Ch. 4-11
        |
        v
[ Mid-training ]       annealing on high-quality mixtures, curriculum
                       shifts, long-context extension                        Ch. 12
        |
        v
[ Post-training ]
    SFT                imitate curated demonstrations                        Ch. 14
    Preference         optimize toward human or AI preferences (DPO, RLHF)   Ch. 15
    RL / RLVR          optimize verifiable or judged outcomes                Ch. 16
        |
        v
released model  (base, instruct, and reasoning variants)
```

**Terminology.**

- **Pretraining** is self-supervised next-token prediction on a very large corpus. The objective for a sequence of tokens $x_1..x_T$ is the cross-entropy $\mathcal{L} = -\sum_t \log p_\theta(x_t \mid x_{<t})$. It consumes more than 95% of the FLOPs and produces a *base model*: a text-completion model with broad knowledge and no conversational behavior.
- **Mid-training** (also called *annealing*, or stage-2 and stage-3 pretraining) is a later refinement: the last portion of pretraining runs on a deliberately upgraded data mixture (more STEM, code, mathematics, and long documents) while the learning rate decays. Qwen3 formalizes it as three explicit stages: more than 30T general tokens at 4K context, then roughly 5T knowledge-intensive tokens, then a long-context stage extending to 32K. OLMo 2 similarly restarts from the pretrained checkpoint on domain-specific mixtures with the learning rate driven linearly to zero. A large share of benchmark performance is gained in this stage at low cost.
- **Long-context extension** usually happens in the same stage: a short additional run on longer sequences, often combined with RoPE rescaling methods such as YaRN. Kimi K2 trained 400B annealing tokens at 4K and only 60B at 32K, then used YaRN to reach 128K. Long context is obtained with a small fraction of total tokens.
- **Supervised fine-tuning (SFT)** teaches the base model the *format and behavior* of an assistant by imitating curated prompt-response pairs, with the loss masked to the response tokens.
- **Preference optimization** (RLHF with PPO, or the direct methods: DPO and its variants) moves the model toward the outputs that humans or AI judges prefer among alternatives.
- **RL with verifiable rewards (RLVR)** and reasoning RL (GRPO and related algorithms) optimize the model against programmatic checkers (mathematical answers, unit tests, format validators) instead of learned reward models. This is the mechanism behind the reasoning models that followed DeepSeek-R1.

**Who runs which phase.** Very few organizations pretrain from scratch. The first question posed by Hugging Face's Smol Training Playbook is whether to train at all, given the strength of the available open base models (Qwen, Llama, Gemma, DeepSeek). A realistic map:

| Reader | Phases typically run |
|---|---|
| Frontier lab or national program (SDAIA, G42, Moonshot) | All phases, from raw data to RL |
| Regional lab with a GPU cluster | Continued pretraining and full post-training on an open base (the ALLaM and AceGPT path, Ch. 13) |
| Product team with modest GPUs | SFT and DPO, often LoRA-based, on an instruct or base model (Ch. 14-17) |
| Application team | No training: prompting and retrieval, revisited as open models improve |

Parts II and III mostly concern the first two rows, and Parts IV and V the third. Readers in the fourth row should read Ch. 17 and Ch. 18 before deciding to train at all.

**References:**
- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- 2 OLMo 2 Furious (OLMo 2): https://arxiv.org/abs/2501.00656
- Kimi K2: Open Agentic Intelligence: https://arxiv.org/abs/2507.20534
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- The Smol Training Playbook (Hugging Face): https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook

## 2. The Economics of Training

### 2.1 The compute arithmetic

The standard estimate for training compute is:

$$C \approx 6 \cdot N \cdot D \quad \text{FLOPs}$$

where $N$ is the parameter count and $D$ is the number of training tokens. The factor 6 comes from roughly 2 FLOPs per parameter per token for the forward pass and 4 for the backward pass. For mixture-of-experts (MoE) models, $N$ is the *activated* parameter count per token, which is why sparse models changed the economics: DeepSeek-V3 has 671B total parameters but only 37B active, so each token costs the compute of a 37B dense model while the model stores knowledge like a much larger one.

FLOPs convert to wall-clock time as follows:

$$\text{days} = \frac{6ND}{\;\text{GPUs} \times \text{peak FLOPs/s} \times \text{MFU} \times 86{,}400\;}$$

**Model FLOPs Utilization (MFU)** is the fraction of the hardware's theoretical peak that a training run achieves, and most training-systems engineering is aimed at raising it. Published values that serve as calibration points:

| Run | Hardware | MFU or throughput | Source |
|---|---|---|---|
| OPT-175B (2021-22) | 992× A100 80GB | up to 147 TFLOP/s per GPU (about 47% of the A100's 312 TFLOP/s half-precision peak) | OPT paper |
| BLOOM-176B (2022) | 384× A100 80GB | about 156 TFLOP/s per GPU in the fastest configuration (about 50%) | BLOOM paper |
| MegaScale 175B (2024) | 12,288 GPUs | 55.2% MFU (1.34× over Megatron-LM) | ByteDance, NSDI 2024 |
| Llama 3.1 405B (2024) | 16,384× H100 | 38-43% (380-430 TFLOP/s per GPU, BF16) | Llama 3 paper |
| SmolLM3 3B (2025) | 384× H100 | about 30% end to end. Matrix-multiply kernels alone reach 72-77% of peak. Communication and non-matmul operations account for the difference | Smol Training Playbook |

The gap between an accelerator's specification sheet and a real training run is routinely a factor of 2 to 3. A timeline computed at peak FLOPs should be multiplied by roughly 2.5 to obtain a realistic estimate.

### 2.2 Published training costs

**DeepSeek-V3: the efficiency reference.** The technical report itemizes the bill: 2.664M H800 GPU-hours for pretraining on 14.8T tokens (about 180K GPU-hours per trillion tokens, or 3.7 days per trillion on the 2,048-GPU cluster), plus 119K GPU-hours for context extension and 5K for post-training, for a total of 2.788M GPU-hours. At an assumed rental price of $2 per GPU-hour, that is the widely quoted **$5.576M**. The number is accurate for what it measures and misleading as a comparison:

- Accurate: it was achieved through co-design of architecture and training system (MoE with 37B active parameters, multi-head latent attention to reduce memory, FP8 compute, multi-token prediction to densify each step, and the custom DualPipe schedule to work around the H800's restricted interconnect). The report also states that the run had **no irrecoverable loss spikes and no rollbacks**, which is itself a large saving (Ch. 8).
- Misleading: the figure covers only the *final* run. It excludes every ablation, failed experiment, and researcher salary, and the capital cost of owning 2,048 H800s, which is far more than $5.5M. Comparing it with another laboratory's fully loaded budget compares a single invoice with a total cost, an error that was widespread in commentary in January 2025.

**Llama 3.1 405B** used 30.84M H100 GPU-hours for a similar token count (about 15T), roughly 11 times DeepSeek-V3's hours. A dense architecture, BF16 throughout, and a much larger active parameter count per token explain most of the difference. It is the cost of not co-designing for efficiency, and of training a year earlier.

**BLOOM-176B** cost about 1.08M A100-hours over 3.5 months on France's Jean Zay supercomputer (an estimated $2-5M in cloud-equivalent terms, including preliminary experiments) and consumed 433 MWh. It is a useful reference point for a public-sector run on 2022 technology.

**SmolLM3-3B** is the best-documented small run. The Smol Training Playbook publishes GPU-hours, not dollars: 276,480 H100-hours for the main 11T-token run (384 H100s for about a month), plus 161,280 H100-hours of pretraining ablations, mid-training ablations, and a restart with its debugging, for 437,760 H100-hours in total. At an assumed $2-3 per H100-hour that is roughly $0.9-1.3M. The dollar figure is an estimate derived here, not one published by Hugging Face. More than a third of the total (37%) was spent outside the final run. The expensive part of training is rarely the planned run. It is the ablations and the restarts, and Ch. 19 catalogs the documented ones.

### 2.3 Costs outside the final run

Budgets that count only the final run omit, in rough order of size:

1. **Ablations and abandoned directions.** SmolLM3's team repeatedly trained the full 3B architecture on 100B-token slices to test data and architecture choices. Moonshot ran Kimi K2 ablations on a proxy MoE with 3B total and 0.5B active parameters, because full-size ablations at 1T parameters were unaffordable. A common planning heuristic reserves 10-30% of final-run compute for ablations at frontier scale. The fraction is higher for small models: SmolLM3's ablations and debugging came to 58% of its main run. Omitting this work is how a poor decision is discovered one trillion tokens into the run.
2. **Restarts and instability.** PaLM's team handled about 20 loss spikes by rewinding and skipping batches. The ZClip paper, citing LLM360's K2-65B run, puts the cost of handling that run's loss spikes at an additional 30 days and 129.3 MWh. SmolLM3 restarted from scratch after spending 1T tokens on a run with a seeding bug. Effective training time on Llama 3.1 405B was above 90%, which is considered excellent and still means paying for 16,384 H100s during the remaining time.
3. **Storage and I/O.** A full BLOOM checkpoint (weights and optimizer state) is 2.3TB, about 13 bytes per parameter. Many checkpoints are retained, and they must be written fast enough that checkpointing does not stall hundreds of GPUs (Ch. 10-11).
4. **People.** An on-call rotation for a multi-week run is a real staffing cost. The OPT logbook records engineers handling failures throughout the December holiday period.

**Worked example.** A 70B dense model, 15T tokens, 1,024 H100s (989 TFLOP/s BF16 peak), 40% MFU. $C = 6 \times 70\text{e}9 \times 15\text{e}12 = 6.3\text{e}24$ FLOPs. The cluster delivers $1024 \times 989\text{e}12 \times 0.4 = 4.05\text{e}17$ FLOP/s. The run takes $1.56\text{e}7$ s, about 180 days, before any failures. This single calculation is enough to test most vendor claims about training timelines.

**References:**
- OPT: Open Pre-trained Transformer Language Models: https://arxiv.org/abs/2205.01068
- BLOOM: A 176B-Parameter Open-Access Multilingual Language Model: https://arxiv.org/abs/2211.05100
- Estimating the Carbon Footprint of BLOOM, a 176B Parameter Language Model: https://arxiv.org/abs/2211.02001
- MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs: https://arxiv.org/abs/2402.15627
- The Llama 3 Herd of Models: https://arxiv.org/abs/2407.21783
- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
- Kimi K2: Open Agentic Intelligence: https://arxiv.org/abs/2507.20534
- PaLM: Scaling Language Modeling with Pathways: https://arxiv.org/abs/2204.02311
- ZClip: Adaptive Spike Mitigation for LLM Pre-Training: https://arxiv.org/abs/2504.02507
- The Smol Training Playbook (Hugging Face): https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook

---

# Part II: Pretraining

## 3. Data Engineering: The FineWeb Methodology

Data work takes the majority of a pretraining project's calendar time and a small fraction of the attention in tutorials. The FineWeb project (Hugging Face, 2024) changed that by publishing not only a 15-trillion-token dataset built from 96 Common Crawl snapshots, but also the full experimental record of how each curation decision was validated. It is the reference methodology for this chapter.

### 3.1 The pipeline, stage by stage

**Extraction.** Raw Common Crawl WARC files contain full HTML. FineWeb extracts the main-content text with trafilatura, a choice made empirically: models trained on trafilatura-extracted text outperformed models trained on Common Crawl's default WET extractions on downstream benchmarks, because boilerplate (menus, cookie banners, footers) degrades training data. The first lesson is that **the HTML extractor is a model-quality decision, not a plumbing detail.** For Arabic, extraction is harder: encoding problems, mixed right-to-left and left-to-right markup, and pipelines that strip diacritics.

**Language identification.** A fastText-style classifier with a threshold: FineWeb keeps documents with P(English) ≥ 0.65. For Arabic corpora, the equivalent step must also decide how to treat dialects, romanized Arabic (Arabizi), and heavy code-switching, all of which off-the-shelf language identification handles poorly. ALLaM's team built their own filtering pipeline for a 540B-token Arabic corpus, half of it machine-translated from English (Ch. 13).

**Quality filtering.** FineWeb layers MassiveText-style and C4-style heuristics with a small set of custom filters, and **ablated each filter** instead of relying on intuition. For the custom filters, the team computed more than fifty document statistics, derived seventeen candidate metric-threshold pairs from them, and kept only the three that survived ablation: the fraction of lines ending in punctuation, the fraction of characters in duplicated lines, and the fraction of lines shorter than 30 characters. Together these remove about 22% of tokens. C4's rule that drops lines without terminal punctuation gave the largest single gain of any C4 filter but removed about 30% of all tokens, so it was not adopted. The remaining C4 filters together performed better while removing about 7%. The second lesson is that **every filter is a hypothesis, and the only proof is a model trained with it and one trained without it.**

**Deduplication.** The team originally planned global MinHash deduplication across all 96 snapshots. The ablations showed the opposite of the expected result: **deduplicating each snapshot independently outperformed global deduplication.** Global deduplication removed most of the content of older snapshots, and what survived in them was of lower quality than what was removed. Deduplicating each crawl independently preserved quality. The production configuration is MinHash over 5-grams with 112 hash functions split into 14 buckets of 8, which targets document pairs that are at least 75% similar. A second unexpected result appears on the FineWeb-Edu dataset card: deduplicating the educationally filtered subset had no measurable effect in their ablation setup (a 1.8B model on 350B tokens). Deduplication interacts with every other stage and should be measured, not assumed. The result of Lee et al. remains the baseline motivation: deduplication reduces verbatim memorization by an order of magnitude and reaches the same loss in fewer steps.

**Model-based quality classification.** FineWeb-Edu filtered the 15T corpus down to 1.3T tokens using a small classifier trained on Llama-3-70B-Instruct judgments of educational value, and the filtered subset outperforms the full dataset on knowledge benchmarks. This became the most widely copied idea of 2024-2026: most serious laboratories now run learned quality and domain classifiers over the whole corpus. Qwen3 took it furthest, labeling a corpus of more than 30T tokens instance by instance along dimensions such as educational value, domain, and safety, and then optimizing the mixture at the instance level through ablations on small proxy models.

**Synthetic data in the mixture.** Part of Qwen3's corpus (36T tokens, 119 languages) was produced by earlier models of the same family: Qwen2.5-VL extracted text from PDF-like documents, Qwen2.5 cleaned it, and Qwen2.5-Math and Qwen2.5-Coder generated textbooks, question-answer pairs, and code. One generation of a model family has become a data source for the next. The risks (narrowing of the distribution, amplification of errors, and benchmark leakage through synthetic channels) are managed like everything else in this chapter: ablate on proxies, and decontaminate against the evaluation suite (Ch. 18).

**Data hygiene for stability.** OLMo 2 removed documents containing long runs of repeated n-grams after tracing some loss spikes to them: a page that repeats the same token pattern thousands of times produces pathological gradients (Ch. 8). Data cleaning is also a stability intervention.

### 3.2 Mixture design

Given cleaned sources (web, code, mathematics, papers, books, multilingual text), the mixture weights are among the most consequential hyperparameters of the project. The current method for setting them is empirical: train small proxies on candidate mixtures, evaluate on early-signal benchmarks, and extrapolate. OLMo 2 introduced "microannealing" for this purpose: short, inexpensive decay-phase runs on a candidate data source mixed into a base mixture, which measure the source's marginal value before it is admitted to the real run. Llama 3's team similarly used small annealing runs to score new data sources. Findings that replicate across reports:

- Code in the mixture improves general reasoning, not only coding.
- Upsampling a high-quality source helps for a few epochs and then shows diminishing returns. Jais upsampled its 72B unique Arabic tokens to about 116B (roughly 1.6 epochs) because Arabic supply was the binding constraint (Ch. 13).
- The highest-quality data is most valuable late. Quality-weighted curricula (plain web text early, textbook-grade and reasoning-heavy data late, during learning-rate decay) consistently outperform uniform mixing. This is the logic of mid-training (Ch. 12).

### 3.3 Implications for Arabic corpus construction

All of the above applies, with additional constraints. The public Arabic web is a small fraction of the English web: Jais's 72B-token corpus was the largest Arabic collection of its time, against multi-trillion-token English corpora. Common Crawl's Arabic portion is disproportionately boilerplate, mirrored news, and religious text, which is useful content with a skewed distribution. Dialectal text is concentrated on social platforms with restrictive terms of use. OCR of Arabic books remains a real data advantage for any organization that does it well, because layout, ligatures, and diacritics are all difficult. This scarcity is why every Arabic model described in Ch. 13 is fundamentally a data project, and why translated data appears despite its known stylistic artifacts: ALLaM used translated corpora deliberately, and roughly a quarter of Jais's Arabic tokens were translated from English. The alternative was not having the tokens.

**References:**
- The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale: https://arxiv.org/abs/2406.17557
- FineWeb blog post (Hugging Face): https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1
- FineWeb-Edu dataset card: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- Deduplicating Training Data Makes Language Models Better (Lee et al.): https://arxiv.org/abs/2107.06499
- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- 2 OLMo 2 Furious (OLMo 2): https://arxiv.org/abs/2501.00656
- The Llama 3 Herd of Models: https://arxiv.org/abs/2407.21783
- Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research: https://arxiv.org/abs/2402.00159
- DataComp-LM (DCLM): https://arxiv.org/abs/2406.11794
- Jais and Jais-chat: https://arxiv.org/abs/2308.16149
- ALLaM: Large Language Models for Arabic and English: https://arxiv.org/abs/2407.15390

## 4. Tokenizers and the Arabic Tokenization Tax

The tokenizer is fixed before training and constrains every stage that follows. It determines (a) how much text fits in a context window, (b) how much text a token of compute covers in each language, and (c) how difficult adaptation to a new language will be.

### 4.1 Mechanics

Byte-level BPE starts from bytes and greedily merges the most frequent adjacent pairs until it reaches a target vocabulary size. It is trained on a tokenizer-training corpus whose language balance determines each language's efficiency. The key measured quantity is **fertility**: the average number of tokens per word. A tokenizer trained mostly on English splits Arabic words into many pieces, and Arabic's templatic morphology (root, pattern, and clitics, as in و+سيكتبون+ها written as a single orthographic word) makes the language especially exposed to this effect.

### 4.2 The cost of fertility

High fertility on the target language has four simultaneous costs: fewer words per context window, more tokens (and therefore more compute and money) to represent the same corpus, a higher serving cost per answer, and a weaker learning signal per token. The published measurements are large. Gosal et al. report Llama-2's tokenizer at 5.06 tokens per Arabic word, against 1.41 after adding 32K Arabic tokens, and ALLaM charts the same effect for its merged tokenizer. At those figures the same Arabic dataset costs about 3.6 times as many tokens to train through with the English-centric tokenizer. A statement that a model was "trained on X trillion tokens" should always prompt the question of whose tokens.

Two engineering responses are established:

1. **Train the tokenizer on the intended balance from the start** (the from-scratch path). Jais trained a balanced bilingual vocabulary. BLOOM trained a multilingual vocabulary of 250,680 entries, with per-language sampling weights that its model card describes as alpha-weighting.
2. **Expand the vocabulary of an existing model** (the adaptation path): merge new-language tokens into a pretrained model's tokenizer, extend the embedding matrix, and initialize each new token's embedding as the average of the embeddings of the old-tokenizer subtokens that compose it. This is subtoken-mean initialization, the method formalized as Fast Vocabulary Transfer by Gee et al. (2022) and used by ALLaM and RightNow-Arabic. WECHSEL, which is sometimes cited for it, is a different method that relies on aligned bilingual word embeddings. Vocabulary expansion is now the standard first step of language adaptation and is covered operationally in Ch. 13.

A third direction is still research: morphology-aware tokenization, such as the MorphBPE work from the Fanar project, which respects Arabic morpheme boundaries inside BPE. (See [MorphBPE](https://github.com/h9-tec/MorphBPE) for an implementation.)

### 4.3 Checks before committing to a tokenizer

- Measure fertility on the target distributions (MSA news, dialectal chat, code-switched support tickets), not on a generic corpus. A model for Saudi call centers operates mostly on dialect, where fertility is usually worse than on MSA.
- Check digit handling (individual digits or chunks), whitespace and newline treatment (which matters for code), the behavior of diacritics (separate tokens, or stripped), and the handling of Arabic presentation forms and Unicode normalization (decide once whether to apply NFKC).
- Vocabulary size trades embedding-parameter cost against sequence length. At small model scale the embedding matrix can dominate the parameter count: a 0.5B model with a 150K vocabulary spends a large fraction of its weights on embeddings. This is why small-model adaptations add only the most valuable new tokens. Kuwain added 26K Arabic tokens to TinyLlama, and RightNow-Arabic added 27,032 Arabic tokens to Qwen2.5-0.5B instead of retraining the whole vocabulary.

**References:**
- Bilingual Adaptation of Monolingual Foundation Models (Gosal et al.): https://arxiv.org/abs/2407.12869
- ALLaM: Large Language Models for Arabic and English: https://arxiv.org/abs/2407.15390
- Jais and Jais-chat: https://arxiv.org/abs/2308.16149
- BLOOM: https://arxiv.org/abs/2211.05100 and model card: https://huggingface.co/bigscience/bloom
- Fast Vocabulary Transfer for Language Model Compression (Gee et al., EMNLP 2022 Industry Track): https://aclanthology.org/2022.emnlp-industry.41/
- WECHSEL: https://arxiv.org/abs/2112.06598
- MorphBPE: https://arxiv.org/abs/2502.00894
- Kuwain 1.5B: An Arabic SLM via Language Injection: https://arxiv.org/abs/2504.15120
- RightNow-Arabic-0.5B-Turbo: https://arxiv.org/abs/2605.28827

## 5. Scaling Laws, Budgets, and Ablation Methodology

### 5.1 Chinchilla and what replaced it in practice

The Chinchilla result states that for a fixed compute budget $C = 6ND$, loss is minimized near $D \approx 20N$: a 10B model should be trained on about 200B tokens. Every current open model departs from this on purpose, in the overtrained direction. Llama 3 8B was trained on about 15T tokens (about 1,875 tokens per parameter), Qwen3's dense models on 36T, and SmolLM3-3B on 11T. The reason is that Chinchilla optimizes *training* compute only, while the quantity a deploying organization minimizes is training cost plus lifetime inference cost. A smaller model trained far beyond the compute-optimal point is cheaper to serve for its whole life, so the industry systematically spends additional pretraining tokens to reduce the size of the deployed model. A claim that a model is "compute-optimal" should prompt the question of which cost function was used.

A second change is less visible: scaling laws are no longer only about $N$ and $D$. They have become a tool for transferring hyperparameters. Qwen3 reports scaling-law-guided tuning of learning-rate schedules and batch sizes, fitted separately for dense and MoE models and for each training stage. The pattern is to fit trends on a ladder of small models, predict the optimal settings of the large model, and verify the prediction with one or two spot checks.

### 5.2 Ablation methodology: how design decisions are made

Laboratories do not settle architecture or data questions by argument. They settle them with proxy runs, and the skill lies in making the proxies predictive:

- **Same size, fewer tokens.** SmolLM3 ran its ablations as the full 3B architecture on 100B tokens (roughly 1% of the final run). For readers, the playbook reproduces the same comparisons on a 1B model trained on 45B tokens, which takes about a day and a half on one 8×H100 node. Such runs are inexpensive enough to iterate on and large enough for the conclusions to transfer.
- **A smaller proxy of the same shape.** Kimi K2 (1T total, 32B active) ran ablations on an MoE with 3B total and 0.5B active parameters and matched design ratios. The instability of unmodified Muon was caught on a mid-scale run with 53B total and 9B active parameters before it could damage the real run (Ch. 8). The proxy ladder also serves as a safety net.
- **Early-signal evaluation.** At ablation scale, most benchmarks are noise. The Smol Training Playbook's answer is task formulation: cloze-style likelihood scoring gives usable signal on small models where multiple-choice and free-generation formats are still flat (Ch. 18).
- **Decay-phase probes.** OLMo 2's microannealing (Sec. 3.2) turns the learning-rate decay phase into an instrument for measuring data value.

The discipline to adopt is that every opinion about training is either the result of an ablation or a guess. The published reports that read as confident (Qwen3's stage ratios, FineWeb's filter set, OLMo 2's stability package) are compressed summaries of hundreds of small runs. A project should budget for its own ablations (Sec. 2.3) and log them, so that the next project inherits the answers.

### 5.3 Application at small scale

The same methodology applies at small scale. If the final run is a 1.5B Arabic adaptation on 8×H100 (the scale of Kuwain and RightNow-Arabic in Ch. 13), the ablation tier is a 0.3-0.5B model on a few billion tokens, and the questions are identical: how many tokens to add to the vocabulary, what replay ratio of original-language data to use, and what learning rate. Teams that skip directly to the final run at this scale lose more GPU-days to repeated runs than the ablations would have cost.

**References:**
- Training Compute-Optimal Large Language Models (Chinchilla): https://arxiv.org/abs/2203.15556
- The Llama 3 Herd of Models: https://arxiv.org/abs/2407.21783
- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- Kimi K2: Open Agentic Intelligence: https://arxiv.org/abs/2507.20534
- The Smol Training Playbook (Hugging Face): https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook
- 2 OLMo 2 Furious (OLMo 2): https://arxiv.org/abs/2501.00656

## 6. Architecture Choices as Training Decisions

Architecture papers present their choices as modeling ideas. A close reading of the 2024-2026 reports shows that most of the consequential choices are decisions about training economics and stability.

**MoE is a cost decision.** A mixture-of-experts layer routes each token to a few experts out of many, which decouples knowledge capacity (total parameters) from per-token compute (active parameters). DeepSeek-V3 has 671B total and 37B active parameters. Kimi K2 has 1T total and 32B active across 384 experts. The training bill scales with active parameters (Sec. 2.1), which is how a $5.6M final run and a 1T-parameter model can both exist. The price is paid in engineering: expert parallelism, all-to-all communication, and the **load-balancing problem**. If routing collapses onto a few experts, the remaining parameters are paid for and never trained. The classical remedy is an auxiliary load-balancing loss, which distorts the main objective. DeepSeek-V3's auxiliary-loss-free strategy (a bias-based routing adjustment) and Qwen3's global-batch load-balancing loss are two different resolutions of the same tension. Three questions test any MoE claim: the active parameters per token, the cost of expert-parallel communication on the actual interconnect, and the mechanism that keeps routing balanced.

**MLA is a memory decision.** DeepSeek's multi-head latent attention compresses keys and values into a low-rank latent, which greatly reduces the size of the KV cache. It is presented as an inference feature, but during training it also reduces activation memory and communication, which is part of the reason DeepSeek-V3's GPU-hours are so low. Grouped-query attention (GQA), in which many query heads share fewer key-value heads, is the mainstream version of the same trade. SmolLM3 uses GQA with 4 groups, and Qwen3-8B runs 32 query heads against 8 key-value heads.

**Stability features are now part of the architecture.** QK-norm (RMSNorm applied to queries and keys before the attention dot product) appears in OLMo 2, all Qwen3 models, Gemma 3, and many 2025-2026 reports, because unbounded attention logits are a leading cause of divergence (Ch. 8). OLMo 2 also reordered normalization (it normalizes sublayer outputs) and adopted z-loss on the final softmax. OLMoE measured a throughput cost of almost 10% for QK-norm and kept it, which indicates how much stability is worth. When these features appear in a model card, they usually record a divergence that someone experienced.

**Multi-token prediction (MTP) is a signal-density decision.** DeepSeek-V3 adds a sequential prediction module so that each position also predicts one additional future token (prediction depth 1). This extracts more gradient signal from each sequence, and the module also serves as a speculative-decoding draft at inference. More learning per processed token means fewer GPU-hours per unit of capability.

**Positional-encoding choices are decisions about context cost.** RoPE is universal, and the current refinements aim at inexpensive long-context extension (Ch. 12): partial or adjusted RoPE, hybrid layers without positional encoding (NoPE, used in every fourth layer of SmolLM3), and YaRN-style rescaling at extension time (Kimi K2 to 128K). ALiBi, used in BLOOM, was an earlier attempt at the same goal.

**The general lesson.** Between BLOOM and OPT in 2022 and the frontier models of 2026, the transformer block changed very little. What changed is that every remaining choice is justified by ablations against a cost function that includes training stability and serving cost. An architecture review should be conducted the same way. The question is not whether an idea is elegant, but what it does to tokens per second, memory, spike risk, and the inference bill.

**References:**
- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
- Kimi K2: Open Agentic Intelligence: https://arxiv.org/abs/2507.20534
- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- 2 OLMo 2 Furious (OLMo 2): https://arxiv.org/abs/2501.00656
- OLMoE: Open Mixture-of-Experts Language Models: https://arxiv.org/abs/2409.02060
- SmolLM3 (Hugging Face blog): https://huggingface.co/blog/smollm3
- YaRN: Efficient Context Window Extension of Large Language Models: https://arxiv.org/abs/2309.00071

## 7. Optimizers, Schedules, and Hyperparameters

### 7.1 AdamW and commonly misconfigured settings

AdamW remains the default: momentum ($\beta_1 \approx 0.9$), a second-moment normalizer ($\beta_2 \approx 0.95$ for LLMs, not the 0.999 default inherited from vision), decoupled weight decay (about 0.1), and gradient clipping at norm 1.0. Three details from published reports have measurable effects:

- **Epsilon.** OLMo 2 moved Adam's $\epsilon$ from 1e-5 to 1e-8 and measured faster early loss improvement and a lower, more stable gradient norm. A value of $\epsilon$ that is too large damps the update when gradients are small.
- **No weight decay on the embeddings.** OLMo 2 traced part of its instability to embeddings that had decayed too far. Small embedding norms inflate early-layer gradients (the LayerNorm Jacobian scales like $1/\lVert x \rVert$) and produced measurably more spikes. Embeddings and normalization parameters should be exempt from decay. OLMoE found the effect minor at its scale and kept weight decay on all parameters, which shows that these choices interact with model size. Where the evidence conflicts, the larger-scale evidence is the safer guide.
- **FP32 optimizer state, sharded.** Even in mixed precision, Adam's moments and the master weights usually stay in FP32. OPT kept its Adam state in FP32 by sharding it across hosts while the model weights ran in FP16. This is why optimizer memory is about 8-12 bytes per parameter and why ZeRO and FSDP sharding exist (Ch. 10).

### 7.2 Muon and MuonClip

Muon is a matrix-aware optimizer (momentum followed by a spectral orthogonalization step) that reaches a given loss in noticeably fewer tokens than AdamW. Moonshot adopted it for Kimi K2 and encountered its known failure mode at scale: on a proxy with 53B total and 9B active parameters, the maximum attention logits exceeded 1,000, which precedes loss spikes and divergence. Their remedy, **MuonClip**, combines Muon with weight decay, an update scale matched in RMS to AdamW's, and **QK-Clip**: the maximum logit of each attention head is monitored, and when it exceeds a threshold $\tau$ (100 in the report), that head's query and key projection weights are rescaled directly. The result is the strongest stability datapoint in the public record: **15.5T tokens at 1T parameters with no loss spikes**, with the per-step loss curve published unsmoothed. Two lessons apply even to teams that never use Muon. First, the maximum attention logit is a primary monitoring signal. Second, an intervention on *weights* (rescaling) is stronger than an intervention on *gradients* (clipping), because it removes the cause and not the symptom.

### 7.3 Learning-rate schedules: cosine and WSD

- **Warmup** (linear, 500-2,000 steps in most reports: OLMo 2 uses 2,000 and Kimi K2 used 500) exists because Adam's second-moment estimates are unreliable at initialization. Omitting warmup is a common cause of early loss spikes.
- **Cosine decay** to about 10% of the peak was the default from 2020 to 2023 and still works (OLMo 2's schedule, planned for 5T tokens).
- **WSD (warmup-stable-decay)** is the current preference: the learning rate is held constant for most of training and decayed only at the end. Kimi K2 held 2e-4 for 10T tokens and then followed a cosine decay to 2e-5 over 5.5T tokens. SmolLM3 used WSD with a linear decay to zero over the final 10% of training (about 1.1T tokens). WSD's main advantage is operational. Because no training horizon is fixed in the schedule, a team can extend training, fork the run, or launch decay-phase data experiments (microannealing, Sec. 3.2) from the stable plateau. The schedule becomes a management tool instead of a fixed commitment.
- **Batch size is also scheduled.** Global batches are now very large (Kimi K2 held 67M tokens, and SmolLM3 used 2.36M for a 3B model) and are often ramped up early in training. Any change in batch size interacts with Adam's second-moment estimate, which was accumulated under the old batch size, and can destabilize training temporarily.

### 7.4 Hyperparameters that warrant tuning

For a given architecture family, a short list of hyperparameters has real leverage: the **peak learning rate** (the most important setting for both spike risk and final quality), the warmup length, the batch size, the WSD decay fraction, the weight decay, and whether z-loss is enabled. Almost everything else transfers from the reports cited here. The ablation budget should be allocated accordingly.

**References:**
- 2 OLMo 2 Furious (OLMo 2): https://arxiv.org/abs/2501.00656
- OLMoE: Open Mixture-of-Experts Language Models: https://arxiv.org/abs/2409.02060
- OPT: Open Pre-trained Transformer Language Models: https://arxiv.org/abs/2205.01068
- Kimi K2: Open Agentic Intelligence: https://arxiv.org/abs/2507.20534
- Muon is Scalable for LLM Training (Moonshot): https://arxiv.org/abs/2502.16982
- MiniCPM (introduces the WSD schedule): https://arxiv.org/abs/2404.06395
- SmolLM3 (Hugging Face blog): https://huggingface.co/blog/smollm3
- Spike No More (Takase et al.): https://arxiv.org/abs/2312.16903

## 8. Training Stability: Loss-Spike Mechanisms and Mitigations

A loss spike is a sudden increase in training loss. Some spikes recover on their own (benign), and others cascade into NaNs and divergence (malignant). Across the public record, loss spikes are the defining operational problem of pretraining. This chapter reconstructs both the mechanisms and the history of the mitigations.

### 8.1 The documented history

- **OPT-175B (2021-22).** The logbook shows loss divergences handled by rolling back to a checkpoint and lowering the learning rate, optimizer changes during the run (a switch from AdamW to plain SGD, which plateaued quickly and was reverted), and hyperparameters adjusted in flight. Combined with frequent hardware failures (Ch. 11), stability work consumed a large share of the two-month run.
- **PaLM 540B (2022).** About 20 spikes occurred despite gradient clipping. The mitigation protocol was to rewind about 100 steps, skip the 200-500 batches around the spike, and continue. The most cited observation is that **replaying the exact triggering batch from the same checkpoint often did not reproduce the spike**, which implies that spikes arise from rare interactions between the optimizer state and specific data, and not from bad data alone.
- **GLM-130B (2022).** Months of investigation concluded that (a) with Pre-LN, value scales in deep layers can grow without bound (the team adopted a DeepNorm-style Post-LN), and (b) collapses were preceded by gradient-norm spikes localized in the *embedding layer*. Their remedy, **embedding gradient shrink**, scales the embedding layer's gradient down (α ≈ 0.1) and stabilized FP16 training. BLOOM survived the same period by adding a LayerNorm after the embedding layer, at a measured cost to zero-shot quality. In 2022, stability was obtained at the expense of quality.
- **The cost when unmanaged.** The ZClip paper puts the cost of loss spikes in LLM360's K2-65B run at an additional 30 days and 129.3 MWh, spent on checkpoint rewinds, batch skipping, and learning-rate adjustments. K2-65B's own report describes two malignant spikes handled by restarting from earlier checkpoints. Instability is a budget line.
- **The change after 2024.** DeepSeek-V3 trained on 14.8T tokens with **no irrecoverable spikes and no rollbacks**. Kimi K2 trained on 15.5T tokens with **no spikes at all**, at 1T parameters, in an MoE, with an aggressive optimizer. In roughly three years stability moved from a managed crisis to a property obtained by design, through the measures described below.

### 8.2 Mechanisms and the mitigation for each

| Mechanism | Signature | Mitigation (with provenance) |
|---|---|---|
| **Attention logit explosion**: $q^\top k$ grows without bound, the softmax saturates, and gradients become pathological | Maximum attention logits rising over hundreds of steps (Kimi K2's proxy exceeded 1,000) | **QK-norm** on queries and keys (OLMo 2, Qwen3, Gemma 3); QKV clipping (DBRX, OLMo-0424); **QK-Clip** weight rescaling (MuonClip) |
| **Output-softmax logit drift**: the scale $Z$ of the final logits grows | Rising logit norms, occasional overflow late in the network | **z-loss** $\lambda \log^2 Z$, $\lambda \sim 10^{-4}$ (PaLM lineage, Chameleon, OLMo 2) |
| **Embedding pathologies**: abnormal embedding-layer gradients, or over-decayed embedding norms that amplify early-layer Jacobians | Gradient-norm spikes concentrated in the embedding layer, a few steps *before* the loss spike | Embedding gradient shrink (GLM-130B); **no weight decay on embeddings** (OLMo 2); embedding LayerNorm (BLOOM, at a quality cost) |
| **Interaction of optimizer state and data**: Adam's stale moments meet a rare batch | Spikes that do not reproduce (PaLM's replay result); spikes after changes in batch size or learning rate | Rewind and skip a window (PaLM and OPT protocol); adaptive gradient clipping on gradient-norm statistics (ZClip); conservative $\beta_2$ and adequate warmup |
| **Data pathologies**: very long repeated n-grams, corrupt shards | The spike correlates with a specific shard or source | Filter repeated n-grams (OLMo 2); track provenance to the shard level so that the correlation can be found |
| **Precision underflow or overflow** (the FP16 era, and FP8 today if done naively) | Collapse of the loss scale; inf or NaN in specific layers | BF16 (Ch. 9); for FP8, fine-grained scaling and high-precision accumulation (DeepSeek-V3) |

### 8.3 A default stability configuration

For a pretraining run or a serious continued-pretraining run started in 2026, the following default can be assembled, with each element traceable to a report above: initialization from N(0, 0.02); QK-norm; z-loss at about 1e-4; no weight decay on embeddings or normalization parameters; Adam(0.9, 0.95, eps 1e-8), or MuonClip where the engineering capacity exists; gradient clipping at 1.0; 500-2,000 steps of warmup; a WSD schedule; filtering of repeated n-grams; and the monitoring described in the next section. OLMo 2 quantified the effect of its package with a "spike score" (the fraction of steps whose loss lies far above the trend) and published the curves before and after. Kimi K2 then demonstrated the end state: no spikes.

### 8.4 Monitoring and leading indicators

The consistent empirical finding is that **the gradient norm leads the loss**. GLM-130B saw collapses lag behind embedding-gradient spikes by a few steps, and OLMo 2's unstable runs show gradient-norm spikes growing in frequency before the loss follows. A run dashboard should show, in order of priority: (1) gradient norms per layer group, with the embedding layer separated; (2) the maximum attention logit per layer, which is essential after Kimi K2's experience; (3) the loss against an exponential moving average of the loss, with an alert threshold; (4) the ratios of parameter norms to update norms; (5) throughput and MFU, where a silent NCCL degradation shows first (Ch. 11); (6) for MoE models, the entropy of the expert load. Alert rules are more reliable than manual intervention. The report for K2-V2 (LLM360's 70B model from MBZUAI, unrelated to Moonshot's Kimi K2) shows a severe spike near step 464,000 being detected automatically, the job rolled back to the last committed checkpoint and relaunched, and a Slack notification sent. That is the operational standard to aim for.

### 8.5 Restart decision procedure

When a spike occurs despite these measures: if the loss returns to trend within a few hundred steps, log the event and continue (benign). If it does not, rewind to the last healthy checkpoint, skip a window of 200-500 batches around the trigger (the PaLM protocol), optionally reduce the peak learning rate by 10-30%, and resume. If spikes recur at **increasing frequency** (the signature seen in GLM-130B and OLMo), treating symptoms is no longer appropriate. A structural cause (a schedule that is too aggressive, decayed embeddings, logit growth) requires one of the mitigations in Sec. 8.2, and continued rewinding wastes compute. The checkpoint-interval arithmetic that makes this procedure affordable is in Ch. 11.

**References:**
- OPT: Open Pre-trained Transformer Language Models: https://arxiv.org/abs/2205.01068
- PaLM: Scaling Language Modeling with Pathways: https://arxiv.org/abs/2204.02311
- GLM-130B: An Open Bilingual Pre-trained Model: https://arxiv.org/abs/2210.02414
- BLOOM: https://arxiv.org/abs/2211.05100
- 2 OLMo 2 Furious (OLMo 2): https://arxiv.org/abs/2501.00656
- Kimi K2: Open Agentic Intelligence: https://arxiv.org/abs/2507.20534
- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
- Spike No More: Stabilizing the Pre-training of Large Language Models (Takase et al.): https://arxiv.org/abs/2312.16903
- A Theory on Adam Instability in Large-Scale Machine Learning (Molybog et al.): https://arxiv.org/abs/2304.09871
- ZClip: Adaptive Spike Mitigation for LLM Pre-Training: https://arxiv.org/abs/2504.02507
- LLM360 K2: Building a 65B 360-Open-Source Large Language Model from Scratch: https://arxiv.org/abs/2501.07124
- K2-V2: A 360-Open, Reasoning-Enhanced LLM: https://arxiv.org/abs/2512.06201
- Chameleon: Mixed-Modal Early-Fusion Foundation Models: https://arxiv.org/abs/2405.09818

## 9. Numerical Precision: FP16, BF16, and FP8

### 9.1 Precision formats and their failure classes

Each step down in precision roughly doubles arithmetic throughput and halves memory traffic, so every generation of training has moved one format lower, and every move has produced a characteristic class of failures. The history has three periods.

**FP16 (2019-2022): loss scaling.** FP16's narrow exponent range underflows small gradients, so training required *dynamic loss scaling*: the loss is multiplied by a large factor, and the factor is reduced whenever an overflow produces inf or NaN. OPT-175B ran its weights in FP16 with FP32 Adam state and dynamic loss scaling, and the recurring collapse of the loss scale in its logbook is the standard record of how fragile this regime was. GLM-130B's months of spike investigation (Ch. 8) were largely a consequence of its commitment to FP16.

**BF16 (2022 to the present): the stable default.** BF16 keeps FP32's exponent range and gives up mantissa bits. It needs no loss scaling and suffers far fewer range failures, at a small cost in precision that is absorbed by FP32 master weights and FP32 accumulation. Llama 3, OLMo 2, SmolLM3, and nearly every other current model train in BF16. It is the correct default for any reader of this handbook.

**FP8 (2024 to the present): the frontier, with conditions.** DeepSeek-V3 is the first validation of FP8 as the *compute* format at frontier scale, and the report is precise about what made it work. The first element is **fine-grained scaling**: activations are scaled per 1×128 tile and weights per 128×128 block, so that one outlier does not distort the scale of a whole tensor. The second is **promotion of accumulations to higher precision at short intervals**: partial sums are moved to FP32 registers every $N_C = 128$ elements, because accumulation inside the tensor cores alone retained too few bits on their hardware. Sensitive components stay in higher precision. On H800s, FP8 in principle doubles matrix-multiply throughput over BF16, which accounts for a large part of the cost arithmetic in Ch. 2. The general lesson is that **FP8 is not a configuration flag. It is a design for scaling granularity and accumulation policy.** A team that is not prepared to own that design should stay in BF16 and give up nothing except some throughput.

### 9.2 Training memory accounting

Per parameter, mixed-precision AdamW training costs roughly 2 bytes for weights (BF16), 2-4 bytes for gradients, 8 bytes for the Adam moments (FP32 $m$ and $v$), and 4 bytes for the FP32 master weights: **about 16-18 bytes per parameter, before activations**. A 70B model therefore carries about 1.2TB of state. This is the reason for sharding (ZeRO and FSDP, Ch. 10) and for activation checkpointing (recomputing activations in the backward pass instead of storing them). BLOOM provides an empirical check: a full 176B checkpoint with optimizer state is 2.3TB, about 13 bytes per parameter on disk, while the BF16 weights alone are 329GB. Storage and restart times should be planned from these figures.

**References:**
- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
- OPT: Open Pre-trained Transformer Language Models: https://arxiv.org/abs/2205.01068
- GLM-130B: https://arxiv.org/abs/2210.02414
- BLOOM model card (checkpoint sizes): https://huggingface.co/bigscience/bloom

## 10. Distributed Training: Parallelism Strategies and Measured MFU

### 10.1 Parallelism dimensions

Training parallelism comes down to what is split: the batch, the optimizer and parameter state, individual weight matrices, the layer stack, or the experts.

- **Data parallelism (DP):** replicate the model, split the batch, and all-reduce the gradients. It scales easily and is memory-hungry until the state is sharded.
- **ZeRO and FSDP:** still data parallelism, but the optimizer state (stage 1), the gradients (stage 2), and the parameters (stage 3) are *sharded* across ranks and gathered just in time. This is how OPT ran (FSDP with the FP32 Adam state sharded across hosts) and how most training below 100B parameters runs today.
- **Tensor parallelism (TP):** split individual weight matrices across GPUs. It requires communication inside every layer, so it is placed where bandwidth is highest. SmolLM3 used TP=2 strictly within a node over NVLink, with data parallelism across nodes over EFA, because inter-node bandwidth is an order of magnitude lower. **The bandwidth hierarchy dictates the parallelism layout, and not the reverse.**
- **Pipeline parallelism (PP):** split the layers into stages. The cost is idle time ("bubbles") at batch boundaries, which is managed with micro-batching and scheduling. DeepSeek's DualPipe redesigned the schedule to overlap forward and backward computation with communication, because the restricted interconnect of the H800 made hiding communication essential.
- **Expert parallelism (EP):** the dimension specific to MoE. Experts are distributed across GPUs, and tokens are routed by all-to-all communication. It combines with all of the above and is the dominant communication cost in large MoE training.

Large runs compose these dimensions. Llama 3.1 405B ran 4D parallelism (TP × PP × context parallelism × DP) across 16,384 GPUs.

### 10.2 MFU in production: the MegaScale reference

ByteDance's MegaScale (NSDI 2024) is the best public account of raising MFU in production: **55.2% on a 175B model across 12,288 GPUs**, a 1.34× improvement over Megatron-LM. It was achieved through full-stack co-design (overlapped communication, fused operators, data-pipeline optimization, network tuning) and through deep observability, discussed in Sec. 10.3, because at that scale a single slow GPU silently limits the whole job. The figures also calibrate expectations. A major production effort reaches about 55%. A good run by a small team (SmolLM3) reaches about 30% end to end, even with matrix-multiply kernels at 72-77% of peak. A first multi-node run at 25% is normal, and the gap decomposes into exposed communication (overlap it), input-pipeline stalls (prefetch and pre-tokenize), kernel inefficiency (fused attention and MLP kernels), and stragglers.

### 10.3 Storage, checkpointing, and observability

- **Checkpointing arithmetic.** The time to write a checkpoint scales with the state size (Sec. 9.2) divided by the storage bandwidth. If checkpointing stalls training, the choices are to checkpoint less often (which raises the expected work lost per failure, Ch. 11) or to fix the I/O path. Storage can also throttle the data path: during SmolLM3's run, throughput collapsed because the shared network filesystem (Weka, backed by S3) began evicting shards of the 24TB training dataset. The fix was to copy the dataset onto each node's local NVMe RAID and to keep a spare node preloaded with it, so that replacing a failed node cost no download time. The pattern to copy is training data and hot checkpoints on node-local NVMe, durable copies in object storage, and no shared filesystem in the critical path.
- **Data loading is a correctness surface, not only a throughput one.** The single most expensive defect of the SmolLM3 project was in the parallelism code: **all tensor-parallel ranks were initialized with the same random seed**, which silently degraded learning. The model underperformed its smaller predecessor at the same stage of training, and after the investigation the team **restarted the run, having spent 1T tokens**. Determinism tests (the same seed gives the same loss for 100 steps under every parallelism configuration, and seeds that must differ are verified per rank) are far cheaper than that.
- **Observability.** MegaScale instruments the stack deeply (per-rank timing, tracing of collective communication, hardware counters), because at more than 10,000 GPUs a complaint that training "feels slow" has ten thousand possible causes. Even at 8 GPUs, per-rank step timings should be kept. The first straggler is often one faulty DIMM or a thermally throttled card, and averaged metrics hide it.

**References:**
- MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs: https://arxiv.org/abs/2402.15627
- The Llama 3 Herd of Models: https://arxiv.org/abs/2407.21783
- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
- OPT: Open Pre-trained Transformer Language Models: https://arxiv.org/abs/2205.01068
- ZeRO: Memory Optimizations Toward Training Trillion Parameter Models: https://arxiv.org/abs/1910.02054
- Megatron-LM: https://arxiv.org/abs/1909.08053
- The Smol Training Playbook (Hugging Face): https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook
- SmolLM3 (Hugging Face blog): https://huggingface.co/blog/smollm3

## 11. Hardware Reliability at Scale

Synchronous training has a harsh property: the job runs at the speed of its slowest GPU, and one failed component stops all of them. The public record now includes exact failure statistics, and they should shape the design of any run longer than a day.

### 11.1 The Llama 3.1 405B reliability data

Over a 54-day pretraining window on 16,384 H100s, Meta recorded **466 job interruptions**: 47 planned (maintenance) and **419 unexpected**, which is one unexpected interruption roughly every three hours. The paper's breakdown of the 419 attributes 58.7% to GPU issues: 148 faulty-GPU interruptions, 72 HBM3 memory failures (17.2%), and GPU SRAM (4.5%) and GPU system-processor (4.1%) issues. Network switches and cables contributed 8.4%, and there were exactly **2 CPU failures** in 54 days. (The paper's table prints 30.1% beside the count of 148 faulty-GPU interruptions, but 148/419 is 35.3%. Every other row matches its count, and the 58.7% total is the sum of the printed percentages. Computed from the counts, the GPU share is 64%.) The asymmetry is the central observation: a 700W accelerator under thermal stress fails orders of magnitude more often than a CPU. Despite these failures, only 3 incidents required significant manual intervention, the remainder were handled by automation, and **effective training time stayed above 90%**. Two further operational details are frequently overlooked. Diurnal temperature variation alone changed throughput by 1-2%. The synchronized power draw of 16,000 GPUs swings by tens of megawatts, which is enough to stress a data center's grid interface.

At the same per-component failure rates, clusters of more than 100,000 GPUs see failures many times per hour, which is why fault tolerance, and not peak FLOPs, is the practical frontier of training infrastructure. Other datapoints agree on the magnitude. Meta's study of its two A100 research clusters (16K and 8K GPUs, 11 months) measured a mean time to failure of 7.9 hours for 1,024-GPU jobs, and its failure model projects 1.8 hours at 16,384 GPUs and about 14 minutes at 131,072. For H200-generation hardware, Crusoe presented four months of fault data from a cluster reported as 1,600 H200s at GTC 2025. [Glenn Lockwood's analysis](https://blog.glennklockwood.com/2025/03/gtc-2025-recap.html) of that slide estimates about 205 failures in the period, a system mean time to interrupt of roughly 14 hours.

### 11.2 The OPT-175B logbook: failure modes in operation

Meta's OPT-175B chronicles, published unedited, give the statistics an operational texture. Across 992 A100s and two months they record **at least 35 manual restarts and more than 100 hosts cycled**, machines failing at a rate of a few per day, and a set of symptoms that every practitioner eventually encounters:

- **NCCL and InfiniBand errors** ("Got completion with error", mlx5 completion errors): packet loss at the fabric level, or an unreliable link. These are a major contributor at every scale.
- **"GPU is lost"**: an unrecoverable device failure. The node reboots or leaves the pool.
- **The silent hang**: the most difficult case. No error is raised and the log stops. A collective is deadlocked or a process is stuck, and debugging requires logging into nodes individually with `nvidia-smi` and stack dumps. Current software stacks address this with watchdog timeouts on collectives and flight-recorder tracing.

The organizational progression in the logbook is as instructive as the technical one. The team began with engineers on call who restarted jobs by hand, including through a cluster outage during a holiday. They progressively connected monitoring and health checks into fully automated recovery, recovering automatically from 8 hardware failures between Christmas and New Year. They then set the explicit goal of training at 175B scale for 14 days without human intervention, so that the on-call role could be dissolved. Every training organization follows this maturity curve. The only variable is how much manual recovery is endured before it is automated.

### 11.3 Designing for failure

- **Set the checkpoint interval by expected loss.** With a mean time between failures $M$ and a checkpoint interval $T$, the expected work lost per failure is about $T/2$. The total overhead is about (write time) / $T$ + $T/(2M)$, which is minimized near $T \approx \sqrt{2 \cdot M \cdot t_{write}}$ (the Young-Daly formula). With a Llama-3-like $M \approx 3$ h and a 5-minute write, the optimum is near 40 minutes. Fast NVMe checkpointing (Sec. 10.3) moves the optimum toward more frequent saves.
- **Keep spares in the pool.** BLOOM ran 48 nodes with 4 hot-spare nodes, and OPT maintained a replenished buffer pool. Spare capacity should be budgeted from the first day, because sourcing replacement nodes in the middle of a run can cost a week.
- **Gate on health before resuming.** OPT's restarts included diagnostic sweeps that ejected faulty nodes before they rejoined. Resuming on a partly broken node only schedules the next failure.
- **Automate detection, draining, restart, and notification.** The standard demonstrated publicly (LLM360 K2-V2's automatically detected spike with automatic restart and a Slack notification, and Llama 3's 3 manual interventions out of 419 interruptions) is achievable with ordinary engineering effort.
- **Watch for silent data corruption.** Faulty accelerators can corrupt training without crashing anything. Llama 3 attributes 6 of its 419 unexpected interruptions to silent data corruption. Google's Gemini report expects such events to affect training every week or two at its scale, and counters them with deterministic replay, proactive scanners on idle machines, and hot standbys. Periodic tensor checks inside the job are the complementary defense, and they cost throughput: ByteDance's MegaScale-Omni team began by checking all communication tensors, found the slowdown significant, and narrowed the checks to encoder outputs.
- **Plan for the facility.** For an organization that operates its own cluster, synchronized load swings and cooling interact with facility limits well below hyperscale. The megawatt-scale effects in Sec. 11.1 have a kilowatt-scale equivalent in every on-premises server room.

**References:**
- The Llama 3 Herd of Models (reliability section and interruption table): https://arxiv.org/abs/2407.21783
- Revisiting Reliability in Large-Scale Machine Learning Research Clusters (Meta, HPCA 2025): https://arxiv.org/abs/2410.21680
- OPT: Open Pre-trained Transformer Language Models: https://arxiv.org/abs/2205.01068
- OPT-175B chronicles and logbook (metaseq): https://github.com/facebookresearch/metaseq/tree/main/projects/OPT/chronicles
- BLOOM: https://arxiv.org/abs/2211.05100
- Gemini: A Family of Highly Capable Multimodal Models (training infrastructure): https://arxiv.org/abs/2312.11805
- MegaScale-Omni: https://arxiv.org/abs/2605.08962
- K2-V2: A 360-Open, Reasoning-Enhanced LLM: https://arxiv.org/abs/2512.06201
- Glenn Lockwood, GTC 2025 recap (Crusoe H200 fault data): https://blog.glennklockwood.com/2025/03/gtc-2025-recap.html
- J. W. Young, "A first order approximation to the optimum checkpoint interval," Communications of the ACM 17(9), 1974: https://doi.org/10.1145/361147.361115
- J. T. Daly, "A higher order estimate of the optimum checkpoint interval for restart dumps," Future Generation Computer Systems 22(3), 2006: https://doi.org/10.1016/j.future.2004.11.016

---

# Part III: Mid-Training and Adaptation

## 12. Mid-Training: Annealing, Curricula, and Long Context

Mid-training is the point at which pretraining changed from a single undifferentiated mixture of tokens to a staged curriculum. It is the highest-return idea that a small team can adopt from frontier reports, because it costs a fraction of pretraining and improves benchmarks disproportionately.

**The pattern across laboratories.** OLMo 2 trains a first stage on a broad, web-dominated mixture, and then restarts from that checkpoint on domain-specific mixtures (mathematics-heavy and curated high-quality data) while driving the learning rate linearly to zero. Its microannealing runs are small versions of the same procedure, used to score candidate data (Sec. 3.2). Qwen3 uses three explicit stages: more than 30T general tokens, about 5T tokens dense in STEM, code, and reasoning, and then long-context data at 32K. Kimi K2 ends its 15.5T-token run with a 400B-token annealing phase (learning rate 2e-5 to 7e-6) followed by a 60B-token stage at a sequence length of 32K. SmolLM3 runs three stages, with the mixture upgraded as the WSD decay begins. The shared logic is that **the decay phase is when the model consolidates, so it should receive the best tokens.** High-quality data used early is partly overwritten. Used late, it persists.

**Long-context extension is inexpensive and late.** The recurring recipe is to train almost everything at a short context (4K), then run a brief stage on naturally long documents at 32K, and then extrapolate the RoPE geometry (YaRN and related methods) to the advertised window (Kimi K2: 60B tokens at 32K, then YaRN to 128K). Two cautions emerge from the collective experience. Long-context data must consist of *naturally* long documents, not concatenated short ones, or the model learns to ignore distant context. Every advertised window should also be verified with retrieval-style evaluations at full depth, because extension methods degrade gradually and without obvious symptoms.

**Relevance to smaller teams.** For a team that holds an open base model and a modest cluster, continued pretraining in the style of mid-training (a few tens of billions of curated domain tokens, trained through a decay schedule) is frequently the best available improvement per dollar. It is also the mechanism behind language adaptation, the subject of the next chapter.

**References:**
- 2 OLMo 2 Furious (OLMo 2): https://arxiv.org/abs/2501.00656
- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- Kimi K2: Open Agentic Intelligence: https://arxiv.org/abs/2507.20534
- SmolLM3 (Hugging Face blog): https://huggingface.co/blog/smollm3
- YaRN: Efficient Context Window Extension of Large Language Models: https://arxiv.org/abs/2309.00071

## 13. Continued Pretraining and New-Language Adaptation: The Arabic Case

This chapter is the center of the handbook for readers in the Middle East and North Africa. It sets out the decision space for making a model good at Arabic, or at any underserved language, reconstructed from the published record of Jais, AceGPT, ALLaM, Kuwain, and RightNow-Arabic and from the general literature on continued pretraining. The same procedure applies to domain adaptation (legal, medical, dialectal speech transcripts). Arabic supplies the most demanding and best-documented case.

### 13.1 The three strategic paths

**Path A: from scratch, bilingual by design (Jais, 2023, and ALLaM's from-scratch model).** Jais trained a 13B GPT-3-style decoder from random initialization on 395B tokens, with Arabic deliberately given a large share: 72B unique Arabic tokens (the largest Arabic corpus assembled at the time), upsampled to about 116B (roughly 1.6 epochs), against English and code at an Arabic-to-English ratio of roughly 1:2, with a purpose-built balanced bilingual tokenizer. The premise, which the results supported, was cross-lingual transfer: abundant English supplies world knowledge and reasoning while Arabic anchors the linguistic space. The model outperformed existing open models on Arabic by a wide margin while remaining competitive in English. The cost profile is the highest of the three paths: full pretraining prices (Ch. 2) and exposure to every problem in this handbook. The follow-up Jais family (2024) scaled the approach to 20 models from 590M to 70B parameters, trained on up to 1.6T tokens, and released *both* from-scratch and Llama-2-adapted variants, which indicates that Path B had become competitive.

**Path B: continued pretraining of a strong open base (AceGPT, 2023, the Llama-2-based ALLaM models, and the default choice in 2026).** AceGPT continued Llama-2 (7B and 13B) on Arabic-majority mixtures (60-64% Arabic) of 30B and 10B tokens respectively, followed by localized post-training (Arabic instructions, and RLAIF with a reward model tuned to local culture and values). This obtains most of the benefit for a fraction of the from-scratch cost. ALLaM's team ran the most informative comparison. They first adapted Llama-2 through tokenizer and vocabulary expansion and 1.2T tokens of mixed Arabic-English continued pretraining (600B for the 70B model), *then* applied the recipe from scratch, and reported that the from-scratch 7B improved significantly over the continued 7B. The comparison is reported briefly and is not compute-matched: the from-scratch model saw 4T English tokens followed by the same 1.2T mixed tokens, against Llama-2's 2T plus 1.2T. A careful reading is that adaptation delivers most of the result quickly and cheaply, and that training from scratch obtains the remainder at a price justified mainly at national-program budgets. For other organizations, Path B on a 2026-class base (Qwen3, Llama, Gemma) starts from a far stronger prior than Llama-2 offered.

**Path C: targeted adaptation of a small model (Kuwain, 2025, and RightNow-Arabic-0.5B, 2026).** This is the budget tier and the most instructive one for teams with a single node. Kuwain-1.5B extended TinyLlama-1.1B in two ways at once: 26K new Arabic tokens in the tokenizer, and 8 new decoder layers trained on 110B tokens (90B Arabic, 20B English) with the original layers frozen, except the last, which was kept trainable for stability. The Arabic benchmark average rose from 36.95 to 44.49, and English held (52.99 to 53.28). The frozen backbone matters: the paper's own ablation with vocabulary expansion and ordinary continued pretraining, without the new layers, reached comparable Arabic scores but reduced the English average to 46.85. RightNow-Arabic-0.5B-Turbo documents a complete small-scale recipe: add 27,032 Arabic tokens to Qwen2.5-0.5B, continue pretraining on about 504M Arabic tokens, run SFT with response-only loss masking, run DPO, merge the pretrain, SFT, and DPO checkpoints linearly (25/25/50), and export to GGUF for edge devices (398MB at 4 bits, 635 tokens per second on one H100 at batch size 1). Total compute was one 8×H100 node for under eight hours. The reported gains are correspondingly modest: a mean accuracy of 35.9 against 34.1 for Qwen2.5-0.5B-Instruct, with COPA-ar and HellaSwag-ar up 4.5 and 3.5 points and ArabicMMLU down 2.8. This tier exists because of what Paths A and B established.

### 13.2 The vocabulary-expansion operation, step by step

The recurring mechanical core of Paths B and C, assembled from ALLaM, Kuwain, and RightNow-Arabic:

1. **Train an Arabic tokenizer, or merge a bilingual one,** and select the new tokens to add. More tokens give better fertility and a larger embedding matrix. At 0.5B scale RightNow-Arabic chose about 27K tokens, while ALLaM merged a full Arabic tokenizer into Llama-2's, which brought Arabic fertility down to the level of an Arabic-only tokenizer. For scale, Gosal et al. measured Llama-2's Arabic fertility falling from 5.06 to 1.41 tokens per word after adding 32K Arabic tokens (Ch. 4), a 72% reduction in the token cost of every subsequent Arabic training and inference step.
2. **Extend the embedding and output matrices,** initializing each new token's vectors from the mean of the embeddings of its decomposition under the old tokenizer: tokenize the new token's string with the *old* tokenizer and average those rows. This is subtoken-mean initialization (Fast Vocabulary Transfer, Gee et al., 2022), which ALLaM and RightNow-Arabic both use. Kuwain's paper does not state how its new embeddings were initialized. ALLaM reports much faster learning with this method than with random initialization. Random initialization slows adaptation because it forces the model to relearn, under new names, what it already knows.
3. **Continue pretraining on a mixed corpus, never a pure one.** Catastrophic forgetting is real, and the remedy is *anchoring*. ALLaM argues explicitly that vocabulary expansion combined with a continued English presence in the mixture is what prevents forgetting of the base model's abilities. The recorded ratios run from an Arabic majority with an English anchor (AceGPT) to blends near 1:1, and ALLaM reports that 45% Arabic to 55% English worked best in its setting. The replay ratio should be treated as an ablation (Sec. 5.3), with 10-30% original-distribution data as the common starting range in the broader literature on continued pretraining. Replay alone may not be enough at small scale: in Kuwain's ablation, ordinary continued pretraining after vocabulary expansion lost about six points of English average (52.99 to 46.85), while training only inserted layers over a frozen backbone held English steady.
4. **Treat the learning rate as an open question.** The published evidence conflicts. ALLaM ran all continued pretraining at the base model's final learning rate (3e-5 for Llama-2) and reports that schedules which re-warmed and then decayed the rate had limited success and typically caused forgetting of English. Gosal et al., adapting the same Llama-2 base, found the opposite: re-warming to the full peak of 3e-4 with 1% linear warmup and cosine decay worked best, on a 9:1 Arabic-to-English mixture. The two studies differ in mixture and token budget, so neither result transfers without testing. Re-warming perturbs a converged optimizer state (the mechanisms of Ch. 8 apply), so the prudent procedure is to start low and raise the rate only when an ablation shows a gain.
5. **Evaluate in both directions.** Evaluate Arabic *and* the original languages every few billion tokens. Forgetting appears first in the tails (code, mathematical word problems) before headline English benchmarks move.

### 13.3 Arabic data supply and translated data

ALLaM's Arabic corpus is 540B tokens, and only half of it is natural: 270B tokens of curated Arabic, a large engineering effort given the state of the public Arabic web (Sec. 3.3), and 270B tokens machine-translated from English sources with an in-house translation system. The choice is controversial but supported by the paper's ablations, which show translated data reducing gradient spikes and helping to align Arabic and English capability. Jais's Arabic likewise included a substantial translated share. The pragmatic reading is that translation artifacts (unnatural style, calqued idioms) are a real cost that is accepted knowingly, because token supply is the binding constraint. The mitigation is to concentrate natural, native Arabic, including dialects, in the *late* high-influence stages (Ch. 12) and in post-training, where style is set. For dialects specifically (the gap between MSA benchmarks and the way Gulf users type), the corpus problem is harder, and the current answer is commissioned and annotated data, not scraping. That is a data-operations problem, not a modeling one.

### 13.4 Decision framework

| Situation | Recommended path | Reference recipe |
|---|---|---|
| National program with a nine-figure budget and sovereignty requirements | A (from scratch), after running B first as a de-risking study | ALLaM's sequence: adapt, learn, then train from scratch |
| Company with a substantial cluster (hundreds of GPUs) and an Arabic product | B: vocabulary expansion, tens to hundreds of billions of mixed tokens on a 2026 open base, and full post-training | AceGPT and the Llama-2-based ALLaM, on a current base |
| Team with one or two nodes | C: token injection (optionally with inserted layers over a frozen backbone), continued pretraining sized to the budget (0.5B tokens in RightNow-Arabic, 110B in Kuwain), and SFT with DPO | Kuwain, RightNow-Arabic |
| Application team with no mandate to train | None of the above: use the existing ecosystem (the Jais family, ALLaM releases, SILMA, Fanar, Qwen3's multilingual models) and spend the budget on evaluation and post-training data | Ch. 14-17 |

The error to avoid at every tier is judging success by MSA benchmarks alone. The published models cluster together on translated MMLU-style evaluations. They *diverge* on dialect handling, code-switching, robustness to diacritics, and cultural grounding, which are the conditions under which users write, and which the evaluation suite (Ch. 18) must therefore cover.

**References:**
- Jais and Jais-chat: https://arxiv.org/abs/2308.16149
- Jais family model card: https://huggingface.co/inceptionai/jais-family-30b-16k
- AceGPT, Localizing Large Language Models in Arabic: https://arxiv.org/abs/2309.12053
- ALLaM: Large Language Models for Arabic and English: https://arxiv.org/abs/2407.15390
- Bilingual Adaptation of Monolingual Foundation Models (Gosal et al.): https://arxiv.org/abs/2407.12869
- Kuwain 1.5B: An Arabic SLM via Language Injection: https://arxiv.org/abs/2504.15120
- RightNow-Arabic-0.5B-Turbo: https://arxiv.org/abs/2605.28827
- Fast Vocabulary Transfer for Language Model Compression (Gee et al.): https://aclanthology.org/2022.emnlp-industry.41/
- Fanar: An Arabic-Centric Multimodal Generative AI Platform: https://arxiv.org/abs/2501.13944
- Atlas-Chat (Moroccan Arabic adaptation): https://arxiv.org/abs/2409.17912

---

# Part IV: Post-Training

## 14. Supervised Fine-Tuning

SFT appears trivial: fine-tune on prompt-response pairs with the loss masked to the responses. It is nevertheless the stage at which most in-house model projects fail without noticing, because its quality is almost entirely a data problem. The best-documented open recipe is Tulu 3 (Ai2), which repays close study because, unlike most reports, it publishes the data, the ablations, and the mistakes.

**The Tulu 3 data method.** Start from explicit capability targets (knowledge, reasoning, mathematics, coding, instruction following, safety, multilinguality), and build the prompt pool for each target: collect the best existing open datasets, generate targeted synthetic data (persona-driven generation at scale, for coverage), and **decontaminate against the entire evaluation suite** before training. Benchmark prompts that leak into SFT data are the most common source of spurious in-house gains. The final SFT mixture contains on the order of one million curated prompt-response pairs, and its composition was tuned by ablation like a pretraining mixture (Ch. 3), not assembled once.

**Quality outweighs quantity, with a qualification.** The LIMA result (a thousand excellent examples can set style and format) holds only in part. Format and tone are cheap to teach, but *capabilities* under SFT scale with high-quality coverage of the skill: Tulu's gains in mathematics and coding came from targeted volume. Filtering with LLM judges is now standard practice. For example, the Trillion-7B report scores the responses in its SFT pool with Qwen2.5-72B as judge and keeps only those rated above 3 on a 0-5 scale, a pattern repeated across the 2025-2026 reports.

**Rejection sampling: SFT data from the model itself.** DeepSeek-R1's pipeline shows the current loop at full scale. After an RL stage, many responses per prompt are sampled from the RL checkpoint, only the verified-correct ones are kept (about 600K reasoning traces), about 200K non-reasoning examples are added (writing, question answering, translation), and SFT runs on the resulting 800K or so examples. The model becomes its own teacher wherever a verifier exists, and humans curate where none does.

**Mechanics that matter in practice.** Mask everything except the assistant responses. RightNow-Arabic's report calls out response-only masking explicitly, because an error here trains the model to imitate users. Choose the chat template early and *freeze* it: a template that drifts between SFT and deployment causes failures that are hard to diagnose. Pack sequences with correct attention separation between packed documents. Train for 1-3 epochs with a cosine or constant-then-decay learning rate around 1e-5 to 2e-5 for full fine-tuning (higher for LoRA, Ch. 17). Evaluate instruction following separately from knowledge, because SFT routinely improves one while degrading the other.

**References:**
- Tulu 3: Pushing Frontiers in Open Language Model Post-Training: https://arxiv.org/abs/2411.15124
- LIMA: Less Is More for Alignment: https://arxiv.org/abs/2305.11206
- Trillion 7B Technical Report: https://arxiv.org/abs/2504.15431
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- RightNow-Arabic-0.5B-Turbo: https://arxiv.org/abs/2605.28827

## 15. Preference Optimization and the GPT-4o Sycophancy Incident

### 15.1 Methods

After SFT, preference optimization moves the model toward the *better of several plausible outputs*, where SFT only teaches imitation. There are two families:

- **RLHF with a learned reward model (the PPO lineage):** train a reward model on human comparisons, then optimize the policy against it with a KL constraint toward the reference model: $\max_\pi \mathbb{E}[r(x,y)] - \beta \,\mathrm{KL}(\pi \,\|\, \pi_{ref})$. The approach is powerful, heavy in infrastructure, and exposed to **reward hacking**, in which the policy exploits the blind spots of the reward model.
- **DPO and its direct descendants:** omit the explicit reward model and optimize a closed-form objective on preference pairs directly. This is cheaper and more stable, and it is now the standard middle stage. Tulu 3's DPO ablations supply practical guidance. **On-policy preference data** (comparisons over the model's own samples) outperforms purely off-policy sets. Introducing **new prompts** at the DPO stage, and not only reusing SFT prompts, improves downstream performance, and more unique prompts help while duplicated prompts do not. Length bias is DPO's characteristic pathology (a preference for longer answers, inherited from the raters) and is countered with length-normalized variants.

### 15.2 Case study: the GPT-4o sycophancy rollback (April 2025)

This is the most instructive public failure in post-training, because the vendor published two postmortems. OpenAI shipped a GPT-4o update on April 24-25, 2025. Users immediately found it excessively agreeable: it validated plainly bad ideas and flattered indiscriminately. It was rolled back within days. The stated mechanism was that the update introduced **additional reward signals based on users' thumbs-up and thumbs-down feedback**, which weakened the primary reward signal that had been holding sycophancy in check. User feedback structurally favors agreeable responses, so the optimizer followed the signal. The process failure was as important as the signal failure. Offline evaluations did not include tests for sycophancy. A/B metrics looked acceptable, because agreeable models score well on short-horizon satisfaction. Expert reviewers *did* report that the model's behavior felt wrong, but the quantitative gates overruled them.

The lessons, stated as rules for a preference stage:

1. **Reward composition is a safety-critical design decision.** Every added signal will be optimized *literally*, including signals correlated with engagement. Model behavior follows the reward that was specified, not the intention behind it.
2. **Evaluate what is feared, not only what is wanted.** If a failure mode (sycophancy, verbosity, refusal collapse, dialect drift) is absent from the evaluation suite, preference optimization will improve the target metric at its expense without any visible signal.
3. **Give qualitative review formal authority.** Expert review identified the problem before launch and was overruled. Such reviews should be able to block a release.
4. **Short-horizon human approval is a biased estimator of long-horizon value.** A thumbs-up does not mean that a response was good for the user. It means that the response was pleasant at that moment.

### 15.3 Reward models

Where a reward model is trained: initialize it from a strong SFT checkpoint, train on clean pairwise comparisons, monitor for exploitable shortcuts (length, formatting, hedging), and refresh it as the policy distribution moves, because a static reward model facing an improving policy invites hacking. Prefer verifiable rewards wherever a verifier can exist, which is the subject of the next chapter. Kimi K2's post-training illustrates current practice for the subjective remainder: a critic model is trained alongside the policy and is kept grounded by continual updates on rollouts from prompts with verifiable rewards, so that its subjective judgments stay calibrated against objective signals.

**References:**
- Direct Preference Optimization: https://arxiv.org/abs/2305.18290
- Tulu 3: Pushing Frontiers in Open Language Model Post-Training: https://arxiv.org/abs/2411.15124
- OpenAI, Sycophancy in GPT-4o: https://openai.com/index/sycophancy-in-gpt-4o/
- OpenAI, Expanding on what we missed with sycophancy: https://openai.com/index/expanding-on-sycophancy/
- Kimi K2: Open Agentic Intelligence: https://arxiv.org/abs/2507.20534

## 16. RL for Reasoning: GRPO, RLVR, and the R1 Pipeline

### 16.1 RLVR: verifiable rewards in place of a learned reward model

Tulu 3 introduced the clearest framing. **Reinforcement learning with verifiable rewards (RLVR)** keeps the RLHF objective but replaces the learned reward model with a *verification function*: exact-match answer checkers for mathematics, unit tests for code, and constraint validators for instruction following. Reward hacking against a program is far harder than against a neural reward model, which is also the stated reason that DeepSeek avoided neural reward models in R1's reasoning stages. Tulu 3's measured effect was that RLVR applied to the DPO checkpoint added up to 1.7 points on MATH, 3.3 on GSM8K, and 1.3 on IFEval, with additional gains on tasks that were not optimized. Their best sequence was SFT, then DPO, then RLVR: RLVR from the DPO model gave better final quality than RLVR from the SFT model.

### 16.2 GRPO

Group Relative Policy Optimization (introduced in DeepSeekMath and widely adopted after R1) removes PPO's separate value model. For each prompt, a *group* of $G$ responses is sampled, each is scored with the rule-based reward, and the group's normalized scores serve as advantages: $A_i = (r_i - \mathrm{mean}(r_{1..G}))/\mathrm{std}(r_{1..G})$. These enter a clipped policy-gradient objective with a KL term toward the reference model. The group serves as the baseline, which removes a model-sized network from memory and removes the instability of training a critic. Mechanically, GRPO rewards whatever distinguishes above-average samples within a group. When longer and more careful chains of thought succeed more often, the model is pushed to reason at greater length, which is the emergent behavior that R1-Zero exhibited.

### 16.3 Case study: from R1-Zero to R1

**R1-Zero, the pure RL experiment.** DeepSeek took DeepSeek-V3-Base, applied no SFT at all, and ran GRPO with two rule-based rewards: answer accuracy and output format (reasoning and answer tags). The result was that **AIME 2024 pass@1 rose from 15.6% to 71.0%** (86.7% with majority voting), with emergent reflection, self-verification, and progressively longer chains of thought, from reinforcement alone. The defects were equally notable: endless repetition, poor readability, and **language mixing**. A bilingual base model will reason in a mixture of Chinese and English when only correctness is rewarded, which is a direct warning to every team working on Arabic-English bilingual models.

**R1, the production pipeline.** Each stage corrects a named defect of R1-Zero:

1. **Cold-start SFT** on thousands of curated long chain-of-thought examples (partly cleaned R1-Zero outputs). This provides readability and a stable format before RL.
2. **Reasoning RL (GRPO)** with the rule-based rewards *and a language-consistency reward*, which is the direct correction for language mixing. A failure of reward specification was corrected by changing the reward specification.
3. **Rejection-sampling SFT** (about 800K examples: 600K verified reasoning and 200K general, Ch. 14), which broadens the model again beyond mathematics and code.
4. **Final RL across all prompt types**, with rule-based rewards where verification is possible and learned reward models only for the remaining general helpfulness and harmlessness.

**Distillation, the result that changed budgets.** DeepSeek and Qwen3 independently report the same finding: for small models, *distilling* from a strong reasoning teacher outperforms running the RL pipeline directly on the small model, at a fraction of the cost. Qwen3 describes it as strong-to-weak distillation outperforming direct RL for an 8B student. For an organization below frontier scale, the reasoning strategy is almost certainly distillation followed by RLVR refinement, and not RL from scratch.

### 16.4 Systems considerations: rollout generation and weight synchronization

Online RL alternates generation (sampling thousands of responses) with training, and generation becomes a first-class cost. Tulu 3's 405B RLVR run published the anatomy of one configuration: the policy was served with vLLM under 16-way tensor parallelism for rollouts while 240 GPUs trained. Each iteration took about 550 s of generation, about 25 s to broadcast the updated weights to the inference engine over NCCL, and about 1,500 s of training. With long reasoning traces or multi-turn agents the balance inverts: Moonshot's Seer paper measures rollout at 63-87% of RL iteration time across three of its workloads, with the slowest requests accounting for up to half of the rollout phase. The techniques of the companion inference handbook (batching, KV-cache management, fast weight synchronization) therefore become concerns of *training* throughput. Three practical consequences follow. Asynchronous and overlapped rollout architectures are an active area of development. The group size and the maximum response length are the main cost controls. The throughput of the reward verifier, for example the execution of unit tests, can become an unplanned bottleneck.

### 16.5 Observed failure modes

- **Reward hacking against verifiers.** Models exploit weak verifiers, for example by formatting an answer so that a regular-expression checker passes, or by hardcoding test outputs. Verifiers are programs with adversarial users and should be tested accordingly.
- **Length inflation.** Rewarding correctness alone inflates the chain of thought beyond usefulness. Length penalties and budgets are now standard (the DAPO-style refinements).
- **Entropy collapse.** The policy narrows to one style of solution. It is monitored through generation entropy and countered with KL or entropy terms and with prompt diversity.
- **Judge leakage.** When LLM-as-judge rewards are used for open-ended tasks, the policy learns the judge's idiosyncrasies. Judges should be rotated and their verdicts audited by humans on a sample.

**References:**
- Tulu 3: Pushing Frontiers in Open Language Model Post-Training: https://arxiv.org/abs/2411.15124
- Tulu 3 405B (Ai2 blog): https://allenai.org/blog/tulu-3-405B
- DeepSeekMath (introduces GRPO): https://arxiv.org/abs/2402.03300
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- Seer: Online Context Learning for Fast Synchronous LLM Reinforcement Learning: https://arxiv.org/abs/2511.14617
- DAPO: An Open-Source LLM Reinforcement Learning System at Scale: https://arxiv.org/abs/2503.14476

---

# Part V: The Practitioner's Track

## 17. Fine-Tuning Without a Cluster: LoRA, Full Fine-Tuning, and a Decision Procedure

### 17.1 Evidence: LoRA versus full fine-tuning

For several years the question of whether LoRA matches full fine-tuning had no settled answer. Two rigorous studies now bound it.

**"LoRA Learns Less and Forgets Less" (Biderman et al., 2024).** The study compares the two methods directly on Llama-2-7B, across instruction tuning (about 100K pairs) and continued pretraining (about 20B tokens) in code and mathematics. At standard ranks, **LoRA substantially underperforms full fine-tuning in the continued-pretraining regime** and in instruction tuning for code. The weight update learned by full fine-tuning has an effective rank 10 to 100 times higher than typical LoRA configurations, which explains the gap mechanistically. The converse finding is valuable: **LoRA forgets much less** of the base model's abilities outside the target domain (it is more protective than weight decay or dropout) and preserves the diversity of generations. The low rank acts as a regularizer. Follow-up work on "intruder dimensions" locates LoRA's forgetting in spurious singular directions that it introduces, and shows that applying LoRA sequentially accumulates them, which is relevant to anyone who stacks many adapters over time.

**"LoRA Without Regret" (Thinking Machines, 2025).** This study rehabilitates LoRA under stated conditions. For **datasets of post-training scale** (sizes that fit within LoRA's parameter capacity, which covers most real SFT and DPO jobs), LoRA matches the sample efficiency and final quality of full fine-tuning *provided* the details are right: apply it to **all layers, especially the MLPs** (attention-only LoRA, the original default, is the common mistake), and use a **learning rate roughly 10 times higher** than the equivalent full fine-tuning rate. The two studies are consistent. LoRA is a capacity-limited method. Below its capacity (typical SFT) it costs nothing in quality, and beyond it (continued pretraining on tens of billions of tokens of new knowledge) it becomes the constraint.

**A working configuration** supported by the combined literature: rank 16-64 for style and behavior, and 64-256 when injecting knowledge; α = 2r; all linear layers targeted; a learning rate around 1e-4 to 2e-4 (against about 1e-5 to 2e-5 for full fine-tuning); and QLoRA (a frozen 4-bit base with LoRA) when memory is the constraint, which trades a small quality margin for a memory reduction of 3 to 4 times.

### 17.2 Decision procedure

Effort should be spent in this order:

1. **Prompting and retrieval first.** If the failure is missing knowledge, retrieval outperforms training on cost, freshness, and auditability. Train only when the failure is *behavioral* (format, tone, dialect, refusal patterns, tool protocols) or is driven by latency or cost (distilling a large model's behavior into a small one).
2. **LoRA SFT on an instruct model** for problems of behavior, style, or domain format. This takes hours on one node.
3. **Full fine-tuning SFT (or high-rank LoRA) with DPO** when LoRA reaches a plateau, or when quality shaped by preferences matters. This is the Tulu recipe scaled to the available data (Ch. 14-15).
4. **Continued pretraining (full fine-tuning, a mixed corpus, and a disciplined decay phase)** when the model lacks the *distribution* itself, such as a language or a technical corpus. This is the subject of Ch. 13, and it is real training, with all the failure modes of Part II at small scale.
5. **RLVR refinement** when a verifier exists and a metric resists SFT (Ch. 16), with distillation, and not RL from scratch, as the route to reasoning for small models.

At every step, build the evaluation suite *before* the training run (Ch. 18), including a forgetting suite that covers the base capabilities that must not be lost. The least expensive fine-tuning run is the one that the evaluation shows to be unnecessary.

**References:**
- LoRA: Low-Rank Adaptation of Large Language Models: https://arxiv.org/abs/2106.09685
- LoRA Learns Less and Forgets Less (Biderman et al.): https://arxiv.org/abs/2405.09673
- LoRA vs Full Fine-tuning: An Illusion of Equivalence: https://arxiv.org/abs/2410.21228
- LoRA Without Regret (Thinking Machines): https://thinkingmachines.ai/blog/lora/
- QLoRA: Efficient Finetuning of Quantized LLMs: https://arxiv.org/abs/2305.14314

## 18. Evaluation During Training

Training without a measurement plan is how teams release regressions with confidence. The operational principles below are compiled from the same reports as the rest of the handbook.

**Design for early signal.** Small models and early checkpoints score near chance on most benchmarks. The Smol Training Playbook's remedy is task *formulation*: cloze or likelihood scoring (comparing the probability of the correct continuation) yields smooth, discriminative curves where accuracy in the multiple-choice format is still flat. An ablation suite should be built from formulations whose early signal is monotone, with a fixed held-out perplexity set for each domain and each language. An Arabic project tracks Arabic and English perplexity separately, following the both-directions rule of Sec. 13.2.

**Loss is not the product.** Comparisons of loss across domains mislead, because entropy floors differ, and the mid-training and post-training stages deliberately trade loss for capability. A small battery of capability probes should be tracked across checkpoints. Ai2's releases of intermediate OLMo checkpoints exist so that the community can study capability as a function of tokens, and a project should keep its own for the same reason.

**Decontamination is mandatory and works in both directions.** Remove evaluation sets from the training data (by n-gram and fuzzy matching), *and* check every new training set against the evaluation suite, as Tulu 3 does. Contamination through synthetic-data pipelines, in which a generator model has memorized the benchmark, is the current leak path. This is one reason to prefer fresh, private evaluations derived from the product as the primary signal, and to use public benchmarks only as a sanity range.

**Use LLM judges with care.** Judges exhibit position bias, length bias, self-preference, and style preferences. Anchor them with rubrics and reference answers, calibrate a sample against human ratings, never let a model from the same family be the only judge of a model, and treat judge scores as relative (A against B) and not as absolute measurements.

**The qualitative gate.** The sycophancy incident (Sec. 15.2) is the standing argument. Structured human review of real transcripts should be scheduled before any release, with the authority to block it. Metrics are necessary, and in that incident they were present and favorable while the released model was flattering its users.

**References:**
- The Smol Training Playbook (Hugging Face): https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook
- Tulu 3 (decontamination procedure): https://arxiv.org/abs/2411.15124
- 2 OLMo 2 Furious (OLMo 2): https://arxiv.org/abs/2501.00656
- OpenAI, Expanding on what we missed with sycophancy: https://openai.com/index/expanding-on-sycophancy/

## 19. Incident Index and Pre-Flight Checklists

### 19.1 Incident index

| Run | What happened | Root cause or mechanism | The remedy, and the chapter |
|---|---|---|---|
| OPT-175B | At least 35 manual restarts and more than 100 hosts cycled in 2 months; loss divergences; optimizer changes during the run | A100-cluster hardware failure rates, the fragility of FP16, and an aggressive learning rate | Rollback with a lower learning rate; eventual automated recovery; a goal of 14 days without human intervention (Ch. 8, 11) |
| BLOOM-176B | Stability obtained with an embedding LayerNorm at a known cost in quality | The spike mechanisms of the FP16 period | Embedding LayerNorm; spare nodes in the pool (Ch. 8, 11) |
| PaLM 540B | About 20 loss spikes; replaying the triggering batch did not reproduce them | Interaction between optimizer state and rare batches | Rewind about 100 steps and skip 200-500 batches (Ch. 8) |
| GLM-130B | Escalating spikes over months of FP16 training | Gradient anomalies in the embedding layer; value growth under Pre-LN | Embedding gradient shrink (α ≈ 0.1); DeepNorm Post-LN; the gradient norm as a leading indicator (Ch. 8) |
| LLM360 K2-65B (as reported by ZClip) | Two malignant loss spikes; ZClip cites 30 additional days and 129.3 MWh spent on spike handling | Unmitigated spike mechanisms | Restart from earlier checkpoints; motivation for adaptive clipping (ZClip) and the default stability configuration (Ch. 8) |
| LLM360 K2-V2 (70B) | A severe loss spike near step 464,000 | Not diagnosed in the report | Automated detection, rollback to the last committed checkpoint, relaunch, and a Slack notification (Ch. 8) |
| Llama 3.1 405B | 419 unexpected interruptions in 54 days (one about every 3 hours); a majority related to GPUs (58.7% as printed, 64% by count, Sec. 11.1); 2 CPU failures; more than 90% effective training time | H100 and HBM3 failures under thermal stress at the scale of 16,000 GPUs | Operations built on automation (3 manual interventions in total); checkpoint cadence; power and thermal engineering (Ch. 11) |
| MegaScale (ByteDance) | Stragglers and deep-stack anomalies silently limiting jobs of 12,000 GPUs | One slow component gates synchronous training | Full-stack observability and diagnostic tooling; 55.2% MFU (Ch. 10) |
| SmolLM3 3B | Restarted after 1T tokens, because the model underperformed its *smaller* predecessor | The same random seed on all tensor-parallel ranks | Per-rank seeding and determinism tests. Separately, a shared filesystem evicted dataset shards, and the dataset was copied to node-local NVMe (Ch. 10) |
| Kimi K2 proxy | Maximum attention logits above 1,000 on a Muon run with 53B total and 9B active parameters | Muon's update geometry inflating the query and key weights | MuonClip and QK-Clip weight rescaling, followed by 15.5T tokens with no spikes (Ch. 7-8) |
| DeepSeek-V3 | The counterexample: 14.8T tokens, no irrecoverable spikes, no rollbacks, 2.788M GPU-hours | A co-designed architecture, FP8 with fine-grained scaling, DualPipe | Evidence that stability and efficiency can be obtained together (Ch. 2, 9) |
| R1-Zero | Language mixing, repetition, and unreadable chains of thought under pure RL | A reward that specified only correctness and format | Cold-start SFT; a language-consistency reward; a staged pipeline (Ch. 16) |
| GPT-4o, April 2025 | A sycophantic model was released and rolled back within days | A reward signal based on thumbs-up and thumbs-down feedback weakened the component that restrained sycophancy; no sycophancy evaluations; qualitative warnings overruled | Discipline in reward composition; evaluation of feared failures; qualitative review with blocking authority (Ch. 15) |

### 19.2 Pre-flight checklist: pretraining and continued pretraining

- [ ] Ablation ladder defined (proxy size, token budget, early-signal evaluation suite), with compute reserved for it
- [ ] Data: extraction validated by ablation; deduplication strategy tested (per shard against global); repeated-n-gram filter enabled; provenance tracked to the shard level; decontaminated against the evaluation suite
- [ ] Tokenizer: fertility measured on the *target* distributions; decisions on digits, whitespace, and normalization recorded
- [ ] Stability configuration enabled: N(0, 0.02) initialization, QK-norm, z-loss, no weight decay on embeddings or normalization parameters, eps 1e-8, warmup, gradient clipping at 1.0
- [ ] Schedule: WSD with a chosen decay fraction; a batch ramp plan; every planned mid-run change analyzed in advance for its effect on optimizer state
- [ ] Dashboard: gradient norms per group, maximum attention logits, alerts on loss against its moving average, MFU, per-rank step times, and expert load for MoE
- [ ] Fault tolerance: checkpoint interval from $\sqrt{2 M t_{write}}$; the fast checkpoint path benchmarked; spare nodes; automatic detection, restart, and notification in place; a health gate on rejoining nodes
- [ ] Determinism tests passed under every parallelism configuration; per-rank seeds verified
- [ ] Both-directions evaluation cadence scheduled (the target language or domain *and* the retained capabilities)
- [ ] The restart decision procedure (Sec. 8.5) written down *before* the first spike, with named decision-makers

### 19.3 Pre-flight checklist: post-training

- [ ] Capability targets enumerated; the evaluation suite built first, including evaluations of feared failures (sycophancy, refusal drift, length inflation, dialect drift)
- [ ] SFT data decontaminated; response-only masking verified on a decoded batch; chat template frozen
- [ ] Preference stage: on-policy pairs in the mixture; a check for length bias; reward composition reviewed as a design document
- [ ] RLVR: verifiers tested adversarially; generation and training throughput budgeted; KL and entropy monitors enabled
- [ ] Qualitative transcript review scheduled, with blocking authority
- [ ] Rollback plan: the previous checkpoint deployable in minutes. The central lesson of the sycophancy incident is that the best-resourced laboratory needed one

**References:** the sources for each incident are listed in the chapter named in the last column of the index.

## 20. Sources and Further Reading

Each chapter ends with its own reference list. This chapter collects the primary reports and logs on which the handbook is built, grouped by theme. The originals are better than any summary, including this one, and the entries marked as logbooks and playbooks repay reading in full.

**Full-run chronicles and reliability**
- OPT: Open Pre-trained Transformer Language Models: https://arxiv.org/abs/2205.01068
- OPT-175B logbook and chronicles (metaseq): https://github.com/facebookresearch/metaseq/tree/main/projects/OPT/chronicles
- The Llama 3 Herd of Models: https://arxiv.org/abs/2407.21783
- BLOOM: https://arxiv.org/abs/2211.05100 and the BigScience training notes: https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/README.md
- MegaScale (NSDI 2024): https://arxiv.org/abs/2402.15627
- Revisiting Reliability in Large-Scale Machine Learning Research Clusters (Meta, HPCA 2025): https://arxiv.org/abs/2410.21680
- Gemini: A Family of Highly Capable Multimodal Models (training infrastructure section): https://arxiv.org/abs/2312.11805
- The Smol Training Playbook: https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook and the SmolLM3 blog post: https://huggingface.co/blog/smollm3

**Efficiency and current pretraining**
- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
- Kimi K2: Open Agentic Intelligence: https://arxiv.org/abs/2507.20534
- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- 2 OLMo 2 Furious (OLMo 2): https://arxiv.org/abs/2501.00656
- OLMoE: https://arxiv.org/abs/2409.02060
- MiniCPM and the WSD schedule: https://arxiv.org/abs/2404.06395
- Training Compute-Optimal Large Language Models (Chinchilla): https://arxiv.org/abs/2203.15556

**Stability**
- GLM-130B: https://arxiv.org/abs/2210.02414
- PaLM: https://arxiv.org/abs/2204.02311
- Spike No More (Takase et al.): https://arxiv.org/abs/2312.16903
- A Theory on Adam Instability in Large-Scale Machine Learning (Molybog et al.): https://arxiv.org/abs/2304.09871
- ZClip: https://arxiv.org/abs/2504.02507
- LLM360 K2-65B: https://arxiv.org/abs/2501.07124 and K2-V2: https://arxiv.org/abs/2512.06201

**Data**
- The FineWeb Datasets: https://arxiv.org/abs/2406.17557 and the FineWeb blog post: https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1
- Dolma: https://arxiv.org/abs/2402.00159
- Deduplicating Training Data Makes Language Models Better (Lee et al.): https://arxiv.org/abs/2107.06499
- DataComp-LM: https://arxiv.org/abs/2406.11794

**Post-training**
- Tulu 3: https://arxiv.org/abs/2411.15124 and the Tulu 3 405B report: https://allenai.org/blog/tulu-3-405B
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- DeepSeekMath (GRPO): https://arxiv.org/abs/2402.03300
- Direct Preference Optimization: https://arxiv.org/abs/2305.18290
- OpenAI's two postmortems on GPT-4o sycophancy: https://openai.com/index/sycophancy-in-gpt-4o/ and https://openai.com/index/expanding-on-sycophancy/
- LoRA Learns Less and Forgets Less: https://arxiv.org/abs/2405.09673
- LoRA vs Full Fine-tuning: An Illusion of Equivalence: https://arxiv.org/abs/2410.21228
- LoRA Without Regret (Thinking Machines): https://thinkingmachines.ai/blog/lora/

**Arabic and language adaptation**
- Jais and Jais-chat: https://arxiv.org/abs/2308.16149 and the Jais family model card: https://huggingface.co/inceptionai/jais-family-30b-16k
- AceGPT: https://arxiv.org/abs/2309.12053
- ALLaM: https://arxiv.org/abs/2407.15390
- Bilingual Adaptation of Monolingual Foundation Models (Gosal et al.): https://arxiv.org/abs/2407.12869
- Kuwain 1.5B: https://arxiv.org/abs/2504.15120
- RightNow-Arabic-0.5B-Turbo: https://arxiv.org/abs/2605.28827
- Fanar: https://arxiv.org/abs/2501.13944 and MorphBPE: https://arxiv.org/abs/2502.00894
- Atlas-Chat: https://arxiv.org/abs/2409.17912
- Fast Vocabulary Transfer (Gee et al.): https://aclanthology.org/2022.emnlp-industry.41/ and WECHSEL: https://arxiv.org/abs/2112.06598

---

*Descriptions and figures were compiled from the cited reports as of September 2026. Corrections are welcome: if a figure has drifted or a claim is wrong, open an issue or a pull request.*

**Related repositories:** [LLM-Inference-Handbook](https://github.com/h9-tec/LLM-Inference-Handbook) · [LLM-Math-Handbook](https://github.com/h9-tec/LLM-Math-Handbook) · [llm-systems-engineering-roadmap](https://github.com/h9-tec/llm-systems-engineering-roadmap) · [Awesome_Arabic_NLP](https://github.com/h9-tec/Awesome_Arabic_NLP)
