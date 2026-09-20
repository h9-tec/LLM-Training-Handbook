# LLM Training Handbook

**A technical reference on how large language models are trained: the mechanisms, the economics, and the operational record of large-scale training runs.**

Most training tutorials stop at the loss function. This handbook takes the opposite approach: every chapter is anchored in **documented training runs**, including the OPT-175B logbook, the Llama 3.1 405B reliability data, DeepSeek-V3's FP8 recipe, Kimi K2's run without loss spikes, OLMo 2's stability investigation, SmolLM3's restart after one trillion tokens, the Tulu 3 and DeepSeek-R1 post-training pipelines, and the Arabic adaptation literature (Jais, AceGPT, ALLaM, Kuwain). A reader who finishes it should be able to read a training log or a technical report critically, and to recognize a divergence, a throughput regression, or a contaminated evaluation from its signature.

It is the training-side companion to the [LLM-Inference-Handbook](https://github.com/h9-tec/LLM-Inference-Handbook) and the [LLM-Math-Handbook](https://github.com/h9-tec/LLM-Math-Handbook). The inference handbook covers how a model is served economically. This one covers how the model came to exist and what failed along the way.

**Audience.** ML engineers, infrastructure engineers, and technical leads who train, fine-tune, or adapt LLMs, or who need to evaluate training reports. Candidates preparing for training and infrastructure interviews will find the standard questions covered with real numbers: most chapters close with a checkpoint question of that kind and a worked answer.

**Prerequisites.** Nothing to install. Familiarity with basic machine learning and the transformer architecture is assumed. The Arabic-specific sections assume no prior knowledge of Arabic NLP.

**Conventions.** Figures are quoted from the cited reports. Where a figure is derived here, the text says so and shows the arithmetic. Each chapter opens with an *Applies to* line naming the readers it serves (the rows of the table in Ch. 1) and ends with a linked reference list. Cross-references use chapter (Ch.) and section (Sec.) numbers. Code listings are minimal Python or PyTorch sketches, tested on small inputs. They show a mechanism and are not production code.

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
17. [Distillation](#17-distillation)
18. [Agentic and Tool-Use Post-Training](#18-agentic-and-tool-use-post-training)

**Part V: The Practitioner's Track**

19. [Fine-Tuning Without a Cluster: LoRA, Full Fine-Tuning, and a Decision Procedure](#19-fine-tuning-without-a-cluster-lora-full-fine-tuning-and-a-decision-procedure)
20. [Evaluation During Training](#20-evaluation-during-training)
21. [Frameworks and Tooling](#21-frameworks-and-tooling)
22. [Incident Index and Pre-Flight Checklists](#22-incident-index-and-pre-flight-checklists)
23. [Sources and Further Reading](#23-sources-and-further-reading)

---

# Part I: The Map

## 1. What "Training" Means in 2026: The Full Pipeline

*Applies to: all readers.*

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
    Distillation       transfer a stronger model's behavior to a smaller one Ch. 17
    Agentic training   tool use and multi-step tasks in environments         Ch. 18
        |
        v
released model  (base, instruct, and reasoning variants)
```

**Terminology.**

- **Pretraining** is self-supervised next-token prediction on a very large corpus. The objective for a sequence of tokens $x_1..x_T$ is the cross-entropy $\mathcal{L} = -\sum_t \log p_\theta(x_t \mid x_{\lt t})$. It consumes more than 95% of the FLOPs and produces a *base model*: a text-completion model with broad knowledge and no conversational behavior.
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
| Product team with modest GPUs | SFT and DPO, often LoRA-based, on an instruct or base model (Ch. 14-19) |
| Application team | No training: prompting and retrieval, revisited as open models improve |

Each chapter opens with an *Applies to* line that names the rows it serves. Readers in the fourth row should read Ch. 19 and Ch. 20 before deciding to train at all.

**Checkpoint question.** A technical report states that a model was "trained on 15T tokens." Which questions establish what that number means?

*Answer.* Four. (1) Which phases it covers: pretraining only, or also mid-training and long-context tokens. (2) Whose tokens: token counts are not comparable across tokenizers, and an English-centric tokenizer can spend three to four times as many tokens on the same Arabic text (Ch. 4). (3) Whether the tokens are unique: Jais reports 72B unique Arabic tokens upsampled to about 116B (Ch. 13). (4) For mixture-of-experts models, which parameter count applies: training compute follows *active* parameters, so 15T tokens cost a 37B-active model far less than a 405B dense model (Ch. 2).

**References:**
- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- 2 OLMo 2 Furious (OLMo 2): https://arxiv.org/abs/2501.00656
- Kimi K2: Open Agentic Intelligence: https://arxiv.org/abs/2507.20534
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- The Smol Training Playbook (Hugging Face): https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook

## 2. The Economics of Training

*Applies to: all readers. Sec. 2.4 is written for product teams and regional labs.*

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

The H100 peak of 989 TFLOP/s used throughout this handbook is half of the 1,979 TFLOP/s that NVIDIA publishes for BF16 Tensor Cores, a figure that assumes 2:4 structured sparsity. PyTorch's torchtitan uses the same 989e12 value in its MFU accounting, and Llama 3's reported 430 TFLOP/s at 43% MFU implies the same peak. The A100 equivalent is 312 TFLOP/s. The estimate reduces to a few lines of code:

```python
def training_estimate(n_params, n_tokens, n_gpus, mfu, peak_flops=989e12, usd_per_gpu_hour=2.0):
    """6ND estimate. For MoE models pass the ACTIVE parameter count."""
    flops = 6 * n_params * n_tokens
    seconds = flops / (n_gpus * peak_flops * mfu)
    gpu_hours = n_gpus * seconds / 3600
    return {"flops": flops, "days": seconds / 86400,
            "gpu_hours": gpu_hours, "usd": gpu_hours * usd_per_gpu_hour}

training_estimate(70e9, 15e12, 1024, 0.40)["days"]   # 180.0
training_estimate(7e9, 10e9, 8, 0.35)["gpu_hours"]   # 337.0
```

### 2.2 Published training costs

**DeepSeek-V3: the efficiency reference.** The technical report itemizes the bill: 2.664M H800 GPU-hours for pretraining on 14.8T tokens (about 180K GPU-hours per trillion tokens, or 3.7 days per trillion on the 2,048-GPU cluster), plus 119K GPU-hours for context extension and 5K for post-training, for a total of 2.788M GPU-hours. At an assumed rental price of $2 per GPU-hour, that is the widely quoted **$5.576M**. The number is accurate for what it measures and misleading as a comparison:

- Accurate: it was achieved through co-design of architecture and training system (MoE with 37B active parameters, multi-head latent attention to reduce memory, FP8 compute, multi-token prediction to densify each step, and the custom DualPipe schedule to work around the H800's restricted interconnect). The report also states that the run had **no irrecoverable loss spikes and no rollbacks**, which is itself a large saving (Ch. 8).
- Misleading: the figure covers only the *final* run. It excludes every ablation, failed experiment, and researcher salary, and the capital cost of owning 2,048 H800s, which is far more than $5.5M. Comparing it with another laboratory's fully loaded budget compares a single invoice with a total cost, an error that was widespread in commentary in January 2025.

**Llama 3.1 405B** used 30.84M H100 GPU-hours for a similar token count (about 15T), roughly 11 times DeepSeek-V3's hours. A dense architecture, BF16 throughout, and a much larger active parameter count per token explain most of the difference. It is the cost of not co-designing for efficiency, and of training a year earlier.

**BLOOM-176B** cost about 1.08M A100-hours over 3.5 months on France's Jean Zay supercomputer (an estimated $2-5M in cloud-equivalent terms, including preliminary experiments) and consumed 433 MWh. It is a useful reference point for a public-sector run on 2022 technology.

**SmolLM3-3B** is the best-documented small run. The Smol Training Playbook publishes GPU-hours, not dollars: 276,480 H100-hours for the main 11T-token run (384 H100s for about a month), plus 161,280 H100-hours of pretraining ablations, mid-training ablations, and a restart with its debugging, for 437,760 H100-hours in total. At an assumed $2-3 per H100-hour that is roughly $0.9-1.3M. The dollar figure is an estimate derived here, not one published by Hugging Face. More than a third of the total (37%) was spent outside the final run. The expensive part of training is rarely the planned run. It is the ablations and the restarts, and Ch. 22 catalogs the documented ones.

### 2.3 Costs outside the final run

Budgets that count only the final run omit, in rough order of size:

1. **Ablations and abandoned directions.** SmolLM3's team repeatedly trained the full 3B architecture on 100B-token slices to test data and architecture choices. Moonshot ran Kimi K2 ablations on a proxy MoE with 3B total and 0.5B active parameters, because full-size ablations at 1T parameters were unaffordable. A common planning heuristic reserves 10-30% of final-run compute for ablations at frontier scale. The fraction is higher for small models: SmolLM3's ablations and debugging came to 58% of its main run. Omitting this work is how a poor decision is discovered one trillion tokens into the run.
2. **Restarts and instability.** PaLM's team handled about 20 loss spikes by rewinding and skipping batches. The ZClip paper, citing LLM360's K2-65B run, puts the cost of handling that run's loss spikes at an additional 30 days and 129.3 MWh. SmolLM3 restarted from scratch after spending 1T tokens on a run with a seeding bug. Effective training time on Llama 3.1 405B was above 90%, which is considered excellent and still means paying for 16,384 H100s during the remaining time.
3. **Storage and I/O.** A full BLOOM checkpoint (weights and optimizer state) is 2.3TB, about 13 bytes per parameter. Many checkpoints are retained, and they must be written fast enough that checkpointing does not stall hundreds of GPUs (Ch. 10-11).
4. **People.** An on-call rotation for a multi-week run is a real staffing cost. The OPT logbook records engineers handling failures throughout the December holiday period.

**Worked example.** A 70B dense model, 15T tokens, 1,024 H100s (989 TFLOP/s BF16 peak), 40% MFU. $C = 6 \times 70\text{e}9 \times 15\text{e}12 = 6.3\text{e}24$ FLOPs. The cluster delivers $1024 \times 989\text{e}12 \times 0.4 = 4.05\text{e}17$ FLOP/s. The run takes $1.56\text{e}7$ s, about 180 days, before any failures. This single calculation is enough to test most vendor claims about training timelines.

### 2.4 Worked examples at small scale

The same arithmetic applies to the jobs most teams run. All three examples assume one 8×H100 node at 35% MFU. Fine-tuning jobs with short sequences and padding often achieve less, so these are lower bounds on time.

**Continued pretraining.** A 7B model on 10B tokens of curated domain or Arabic text: $C = 6 \times 7\text{e}9 \times 10\text{e}9 = 4.2\text{e}20$ FLOPs. The node delivers $8 \times 989\text{e}12 \times 0.35 = 2.77\text{e}15$ FLOP/s, so the run takes $1.52\text{e}5$ s, about 42 hours, or 337 GPU-hours. At 2 to 3 dollars per GPU-hour the final run costs roughly 700 to 1,000 dollars. The ablations that decide the mixture and learning rate (Sec. 5.3) will usually cost more than that.

**Full-parameter SFT.** An 8B model on one million examples averaging 1,000 tokens, for two epochs, is 2B tokens: $C = 9.6\text{e}19$ FLOPs, about 9.6 hours on the node, or 77 GPU-hours.

**LoRA SFT.** LoRA skips the weight-gradient computation for the frozen base matrices, so a pass costs slightly more than two thirds of the full fine-tuning FLOPs (Sec. 19.1). The same job takes about 6.4 hours. The larger saving is memory, not time.

For scale, Kimi K2's pretraining is $6 \times 32\text{e}9 \times 15.5\text{e}12 \approx 3.0\text{e}24$ FLOPs, and Llama 3.1 405B's is about $3.8\text{e}25$, some 13 times more for a similar token count. Both are four to five orders of magnitude above the jobs in this section.

**Checkpoint question.** Why is DeepSeek-V3's $5.576M not evidence that a frontier model can be built for under $6M?

*Answer.* The figure is GPU-hours for one successful final run multiplied by an assumed rental rate. It excludes the ablation ladder that made the run safe, failed and exploratory runs, staff, data acquisition and processing, and the capital cost of the cluster. SmolLM3's public accounting shows the pattern at small scale: 37% of all GPU-hours were spent outside the final run. A defensible statement is that DeepSeek-V3's *marginal* final-run cost was about one eleventh of Llama 3.1 405B's GPU-hours for a similar token budget, because of MoE sparsity, FP8, and communication co-design.

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
- torchtitan, peak FLOPs table: https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/utils.py
- LoRA Without Regret (Thinking Machines): https://thinkingmachines.ai/blog/lora/

---

# Part II: Pretraining

## 3. Data Engineering: The FineWeb Methodology

*Applies to: frontier labs and regional labs. Product teams assembling a continued-pretraining corpus need Sec. 3.1 and Sec. 3.3.*

Data work takes the majority of a pretraining project's calendar time and a small fraction of the attention in tutorials. The FineWeb project (Hugging Face, 2024) changed that by publishing not only a 15-trillion-token dataset built from 96 Common Crawl snapshots, but also the full experimental record of how each curation decision was validated. It is the reference methodology for this chapter.

### 3.1 The pipeline, stage by stage

**Extraction.** Raw Common Crawl WARC files contain full HTML. FineWeb extracts the main-content text with trafilatura, a choice made empirically: models trained on trafilatura-extracted text outperformed models trained on Common Crawl's default WET extractions on downstream benchmarks, because boilerplate (menus, cookie banners, footers) degrades training data. The first lesson is that **the HTML extractor is a model-quality decision, not a plumbing detail.** For Arabic, extraction is harder: encoding problems, mixed right-to-left and left-to-right markup, and pipelines that strip diacritics.

**Language identification.** A fastText-style classifier with a threshold: FineWeb keeps documents with P(English) ≥ 0.65. For Arabic corpora, the equivalent step must also decide how to treat dialects, romanized Arabic (Arabizi), and heavy code-switching, all of which off-the-shelf language identification handles poorly. ALLaM's team built their own filtering pipeline for a 540B-token Arabic corpus, half of it machine-translated from English (Ch. 13).

**Quality filtering.** FineWeb layers MassiveText-style and C4-style heuristics with a small set of custom filters, and **ablated each filter** instead of relying on intuition. For the custom filters, the team computed more than fifty document statistics, derived seventeen candidate metric-threshold pairs from them, and kept only the three that survived ablation: the fraction of lines ending in punctuation, the fraction of characters in duplicated lines, and the fraction of lines shorter than 30 characters. Together these remove about 22% of tokens. C4's rule that drops lines without terminal punctuation gave the largest single gain of any C4 filter but removed about 30% of all tokens, so it was not adopted. The remaining C4 filters together performed better while removing about 7%. The second lesson is that **every filter is a hypothesis, and the only proof is a model trained with it and one trained without it.**

**Deduplication.** The team originally planned global MinHash deduplication across all 96 snapshots. The ablations showed the opposite of the expected result: **deduplicating each snapshot independently outperformed global deduplication.** Global deduplication removed most of the content of older snapshots, and what survived in them was of lower quality than what was removed. Deduplicating each crawl independently preserved quality. The production configuration is MinHash over 5-grams with 112 hash functions split into 14 buckets of 8, which targets document pairs that are at least 75% similar. A second unexpected result appears on the FineWeb-Edu dataset card: deduplicating the educationally filtered subset had no measurable effect in their ablation setup (a 1.8B model on 350B tokens). Deduplication interacts with every other stage and should be measured, not assumed. The result of Lee et al. remains the baseline motivation: deduplication reduces verbatim memorization by an order of magnitude and reaches the same loss in fewer steps.

The detail behind the deduplication result is instructive. Iterative global deduplication, applied from the newest snapshot to the oldest, left about 4T tokens and removed more than 90% of the oldest snapshots. In the 2013-48 snapshot it removed 94% of the tokens (about 490B down to about 31B), and a model trained on the surviving 31B performed *worse* than one trained on the removed data, which on inspection contained fewer advertisements, keyword lists, and badly formatted pages. Per-snapshot deduplication kept about 20T tokens and matched the RefinedWeb baseline. The team's hypothesis is that the benefit of deduplication comes from removing very large clusters of duplicates, and that removing small clusters across snapshots discards good documents.

The MinHash configuration determines which pairs are caught. With $b$ buckets of $r$ hashes each, two documents with Jaccard similarity $s$ are flagged with probability $1 - (1 - s^r)^b$:

```python
def minhash_match_probability(similarity, hashes_per_bucket=8, buckets=14):
    """Probability that two documents collide in at least one LSH bucket."""
    return 1 - (1 - similarity ** hashes_per_bucket) ** buckets

for s in (0.70, 0.75, 0.80, 0.85):
    print(s, round(minhash_match_probability(s), 3))
# 0.70 -> 0.565, 0.75 -> 0.772, 0.80 -> 0.924, 0.85 -> 0.988
```

These match the probabilities quoted in the FineWeb write-up (56%, 77%, 92%, 98.8%). The curve is a soft threshold: a pair at 70% similarity is still flagged more than half the time, and a pair at 80% escapes about one time in thirteen.

The ablation protocol behind every decision in this section is also worth copying. Each variant was tested by training a model of about 1.8B parameters (the paper reports 1.71B) on about 28B tokens, with two runs per variant on different data samples and seeds, and the most important comparisons were confirmed at 350B tokens. The benchmarks were chosen for early signal: low variance between runs, scores that rise monotonically during training, and scores clearly above the random baseline (Sec. 20.1).

**Model-based quality classification.** FineWeb-Edu filtered the 15T corpus down to 1.3T tokens using a small classifier trained on Llama-3-70B-Instruct judgments of educational value, and the filtered subset outperforms the full dataset on knowledge benchmarks. This became the most widely copied idea of 2024-2026: most serious laboratories now run learned quality and domain classifiers over the whole corpus. Qwen3 took it furthest, labeling a corpus of more than 30T tokens instance by instance along dimensions such as educational value, domain, and safety, and then optimizing the mixture at the instance level through ablations on small proxy models.

**Synthetic data in the mixture.** Part of Qwen3's corpus (36T tokens, 119 languages) was produced by earlier models of the same family: Qwen2.5-VL extracted text from PDF-like documents, Qwen2.5 cleaned it, and Qwen2.5-Math and Qwen2.5-Coder generated textbooks, question-answer pairs, and code. One generation of a model family has become a data source for the next. The risks (narrowing of the distribution, amplification of errors, and benchmark leakage through synthetic channels) are managed like everything else in this chapter: ablate on proxies, and decontaminate against the evaluation suite (Ch. 20).

**Data hygiene for stability.** OLMo 2 removed documents containing long runs of repeated n-grams after tracing some loss spikes to them: a page that repeats the same token pattern thousands of times produces pathological gradients (Ch. 8). Data cleaning is also a stability intervention.

### 3.2 Mixture design

Given cleaned sources (web, code, mathematics, papers, books, multilingual text), the mixture weights are among the most consequential hyperparameters of the project. The current method for setting them is empirical: train small proxies on candidate mixtures, evaluate on early-signal benchmarks, and extrapolate. OLMo 2 introduced "microannealing" for this purpose: short, inexpensive decay-phase runs on a candidate data source mixed into a base mixture, which measure the source's marginal value before it is admitted to the real run. Llama 3's team similarly used small annealing runs to score new data sources. Findings that replicate across reports:

- Code in the mixture improves general reasoning, not only coding.
- Upsampling a high-quality source helps for a few epochs and then shows diminishing returns. Jais upsampled its 72B unique Arabic tokens to about 116B (roughly 1.6 epochs) because Arabic supply was the binding constraint (Ch. 13).
- The highest-quality data is most valuable late. Quality-weighted curricula (plain web text early, textbook-grade and reasoning-heavy data late, during learning-rate decay) consistently outperform uniform mixing. This is the logic of mid-training (Ch. 12).

### 3.3 Implications for Arabic corpus construction

All of the above applies, with additional constraints. The public Arabic web is a small fraction of the English web: Jais's 72B-token corpus was the largest Arabic collection of its time, against multi-trillion-token English corpora. Common Crawl's Arabic portion is disproportionately boilerplate, mirrored news, and religious text, which is useful content with a skewed distribution. Dialectal text is concentrated on social platforms with restrictive terms of use. OCR of Arabic books remains a real data advantage for any organization that does it well, because layout, ligatures, and diacritics are all difficult. This scarcity is why every Arabic model described in Ch. 13 is fundamentally a data project, and why translated data appears despite its known stylistic artifacts: ALLaM used translated corpora deliberately, and roughly a quarter of Jais's Arabic tokens were translated from English. The alternative was not having the tokens.

**Checkpoint question.** Why did per-snapshot deduplication outperform global deduplication in FineWeb, and under FineWeb's MinHash settings, how likely is a pair of documents at 80% similarity to be flagged?

*Answer.* Global deduplication keeps one copy of each duplicate cluster across all 96 snapshots, which removes more than 90% of the oldest snapshots. The documents that survive there are the ones never seen again, which are disproportionately low-quality pages, and a model trained on them was worse than one trained on the removed data. Per-snapshot deduplication removes the large duplicate clusters, which is where the benefit lies, without this selection effect. With 14 buckets of 8 hashes, a pair at similarity 0.8 is flagged with probability $1 - (1 - 0.8^8)^{14} \approx 92\%$.

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

*Applies to: all readers who train or adapt models for languages other than English.*

The tokenizer is fixed before training and constrains every stage that follows. It determines (a) how much text fits in a context window, (b) how much text a token of compute covers in each language, and (c) how difficult adaptation to a new language will be.

### 4.1 Mechanics

Byte-level BPE starts from bytes and greedily merges the most frequent adjacent pairs until it reaches a target vocabulary size. It is trained on a tokenizer-training corpus whose language balance determines each language's efficiency. The key measured quantity is **fertility**: the average number of tokens per word. A tokenizer trained mostly on English splits Arabic words into many pieces, and Arabic's templatic morphology (root, pattern, and clitics, as in و+سيكتبون+ها written as a single orthographic word) makes the language especially exposed to this effect.

### 4.2 The cost of fertility

High fertility on the target language has four simultaneous costs: fewer words per context window, more tokens (and therefore more compute and money) to represent the same corpus, a higher serving cost per answer, and a weaker learning signal per token. The published measurements are large. Gosal et al. report Llama-2's tokenizer at 5.06 tokens per Arabic word, against 1.41 after adding 32K Arabic tokens, and ALLaM charts the same effect for its merged tokenizer. At those figures the same Arabic dataset costs about 3.6 times as many tokens to train through with the English-centric tokenizer. A statement that a model was "trained on X trillion tokens" should always prompt the question of whose tokens.

**Worked example.** A 10-billion-word Arabic corpus is 50.6B tokens at fertility 5.06 and 14.1B tokens at fertility 1.41. One epoch of continued pretraining for a 7B model costs $6 \times 7\text{e}9 \times 50.6\text{e}9 = 2.1\text{e}21$ FLOPs in the first case and $5.9\text{e}20$ in the second. On one 8×H100 node at 35% MFU (Sec. 2.4) that is 8.9 days against 2.5 days for the same text. The context window shrinks in the same proportion: 8,192 tokens hold about 1,600 Arabic words at fertility 5.06 and about 5,800 at 1.41. Every downstream number (training cost, serving cost, latency per word) carries the same multiplier, which the companion inference handbook develops in its chapter on Arabic inference economics.

Two engineering responses are established:

1. **Train the tokenizer on the intended balance from the start** (the from-scratch path). Jais trained a balanced bilingual vocabulary. BLOOM trained a multilingual vocabulary of 250,680 entries, with per-language sampling weights that its model card describes as alpha-weighting.
2. **Expand the vocabulary of an existing model** (the adaptation path): merge new-language tokens into a pretrained model's tokenizer, extend the embedding matrix, and initialize each new token's embedding as the average of the embeddings of the old-tokenizer subtokens that compose it. This is subtoken-mean initialization, the method formalized as Fast Vocabulary Transfer by Gee et al. (2022) and used by ALLaM and RightNow-Arabic. WECHSEL, which is sometimes cited for it, is a different method that relies on aligned bilingual word embeddings. Vocabulary expansion is now the standard first step of language adaptation and is covered operationally in Ch. 13.

A third direction is still research: morphology-aware tokenization, such as the MorphBPE work from the Fanar project, which respects Arabic morpheme boundaries inside BPE. (See [MorphBPE](https://github.com/h9-tec/MorphBPE) for an implementation.)

### 4.3 Checks before committing to a tokenizer

- Measure fertility on the target distributions (MSA news, dialectal chat, code-switched support tickets), not on a generic corpus. A model for Saudi call centers operates mostly on dialect, where fertility is usually worse than on MSA.
- Check digit handling (individual digits or chunks), whitespace and newline treatment (which matters for code), the behavior of diacritics (separate tokens, or stripped), and the handling of Arabic presentation forms and Unicode normalization (decide once whether to apply NFKC).
- Vocabulary size trades embedding-parameter cost against sequence length. At small model scale the embedding matrix can dominate the parameter count: a 0.5B model with a 150K vocabulary spends a large fraction of its weights on embeddings. This is why small-model adaptations add only the most valuable new tokens. Kuwain added 26K Arabic tokens to TinyLlama, and RightNow-Arabic added 27,032 Arabic tokens to Qwen2.5-0.5B instead of retraining the whole vocabulary.

Fertility is simple to measure, and the measurement should be part of every model selection:

```python
def fertility(tokenizer, texts):
    """Average tokens per whitespace-delimited word over a list of strings."""
    n_tokens = sum(len(tokenizer.encode(t, add_special_tokens=False)) for t in texts)
    n_words = sum(len(t.split()) for t in texts)
    return n_tokens / n_words

# from transformers import AutoTokenizer
# tok = AutoTokenizer.from_pretrained("<model>")
# fertility(tok, msa_sample), fertility(tok, gulf_dialect_sample), fertility(tok, code_switched_sample)
```

Whitespace splitting undercounts Arabic morphological words, because clitics attach to their hosts, but it is the convention used in the published comparisons and it is adequate for ranking tokenizers on the same text. Report the figure separately for each register the product serves.

**Checkpoint question.** A vendor states that its Arabic model was "trained on 2T Arabic tokens." What has to be established before the figure can be compared with another model's?

*Answer.* (1) The tokenizer's fertility on Arabic: at 5.06 tokens per word, 2T tokens is about 395B words, while at 1.41 it is about 1.4T words. (2) How much of the text is natural and how much is machine-translated: half of ALLaM's 540B Arabic tokens are translated. (3) How many tokens are unique and how many are repeated epochs: Jais's 116B Arabic tokens are 72B unique tokens upsampled 1.6 times. (4) The register mix: MSA, dialects, and code-switched text have different fertility and different value for a given product.

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

*Applies to: frontier labs and regional labs. Sec. 5.3 applies to any team planning a continued-pretraining run.*

### 5.1 Chinchilla and what replaced it in practice

The Chinchilla result states that for a fixed compute budget $C = 6ND$, loss is minimized near $D \approx 20N$: a 10B model should be trained on about 200B tokens. Every current open model departs from this on purpose, in the overtrained direction. Llama 3 8B was trained on about 15T tokens (about 1,875 tokens per parameter), Qwen3's dense models on 36T, and SmolLM3-3B on 11T. The reason is that Chinchilla optimizes *training* compute only, while the quantity a deploying organization minimizes is training cost plus lifetime inference cost. A smaller model trained far beyond the compute-optimal point is cheaper to serve for its whole life, so the industry systematically spends additional pretraining tokens to reduce the size of the deployed model. A claim that a model is "compute-optimal" should prompt the question of which cost function was used.

A second change is less visible: scaling laws are no longer only about $N$ and $D$. They have become a tool for transferring hyperparameters. Qwen3 reports scaling-law-guided tuning of learning-rate schedules and batch sizes, fitted separately for dense and MoE models and for each training stage. The pattern is to fit trends on a ladder of small models, predict the optimal settings of the large model, and verify the prediction with one or two spot checks.

### 5.2 Ablation methodology: how design decisions are made

Laboratories do not settle architecture or data questions by argument. They settle them with proxy runs, and the skill lies in making the proxies predictive:

- **Same size, fewer tokens.** SmolLM3 ran its ablations as the full 3B architecture on 100B tokens (roughly 1% of the final run). For readers, the playbook reproduces the same comparisons on a 1B model trained on 45B tokens, which takes about a day and a half on one 8×H100 node. Such runs are inexpensive enough to iterate on and large enough for the conclusions to transfer.
- **A smaller proxy of the same shape.** Kimi K2 (1T total, 32B active) ran ablations on an MoE with 3B total and 0.5B active parameters and matched design ratios. The instability of unmodified Muon was caught on a mid-scale run with 53B total and 9B active parameters before it could damage the real run (Ch. 8). The proxy ladder also serves as a safety net.
- **Early-signal evaluation.** At ablation scale, most benchmarks are noise. The Smol Training Playbook's answer is task formulation: cloze-style likelihood scoring gives usable signal on small models where multiple-choice and free-generation formats are still flat (Ch. 20).
- **Decay-phase probes.** OLMo 2's microannealing (Sec. 3.2) turns the learning-rate decay phase into an instrument for measuring data value.
- **Small-scale ranking is predictive.** Ai2's DataDecide study trained 1,050 models across 25 corpora and 14 sizes, and found that ranking candidate datasets at a single small size such as 150M parameters predicts the better dataset at 1B in about 80% of pairwise comparisons. None of eight scaling-law extrapolation methods did better for the same compute. The practical conclusion is that a well-chosen small proxy is usually sufficient for data decisions, and that the choice of evaluation metric matters more than the sophistication of the extrapolation.

The discipline to adopt is that every opinion about training is either the result of an ablation or a guess. The published reports that read as confident (Qwen3's stage ratios, FineWeb's filter set, OLMo 2's stability package) are compressed summaries of hundreds of small runs. A project should budget for its own ablations (Sec. 2.3) and log them, so that the next project inherits the answers.

### 5.3 Application at small scale

The same methodology applies at small scale. If the final run is a 1.5B Arabic adaptation on 8×H100 (the scale of Kuwain and RightNow-Arabic in Ch. 13), the ablation tier is a 0.3-0.5B model on a few billion tokens, and the questions are identical: how many tokens to add to the vocabulary, what replay ratio of original-language data to use, and what learning rate. Teams that skip directly to the final run at this scale lose more GPU-days to repeated runs than the ablations would have cost.

**Checkpoint question.** With a budget of $10^{22}$ FLOPs, what model size and token count does Chinchilla suggest, and what is a reasonable alternative for a model that will be served widely?

*Answer.* With $C = 6ND$ and $D = 20N$, $C = 120N^2$, so $N = \sqrt{10^{22}/120} \approx 9.1\text{B}$ parameters and $D \approx 183\text{B}$ tokens. For a widely served model the inference bill dominates, so a smaller, overtrained model is the usual choice: a 3B model absorbs the same budget with $D = 10^{22} / (6 \times 3\text{e}9) \approx 556\text{B}$ tokens, about 185 tokens per parameter. It gives up some loss at this training budget and costs about a third as much per generated token for the rest of its life.

**References:**
- Training Compute-Optimal Large Language Models (Chinchilla): https://arxiv.org/abs/2203.15556
- The Llama 3 Herd of Models: https://arxiv.org/abs/2407.21783
- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- Kimi K2: Open Agentic Intelligence: https://arxiv.org/abs/2507.20534
- The Smol Training Playbook (Hugging Face): https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook
- 2 OLMo 2 Furious (OLMo 2): https://arxiv.org/abs/2501.00656
- DataDecide: How to Predict Best Pretraining Data with Small Experiments: https://arxiv.org/abs/2504.11393

## 6. Architecture Choices as Training Decisions

*Applies to: frontier labs and regional labs, and any reader who evaluates model cards.*

Architecture papers present their choices as modeling ideas. A close reading of the 2024-2026 reports shows that most of the consequential choices are decisions about training economics and stability.

**MoE is a cost decision.** A mixture-of-experts layer routes each token to a few experts out of many, which decouples knowledge capacity (total parameters) from per-token compute (active parameters). DeepSeek-V3 has 671B total and 37B active parameters. Kimi K2 has 1T total and 32B active across 384 experts. The training bill scales with active parameters (Sec. 2.1), which is how a $5.6M final run and a 1T-parameter model can both exist. The price is paid in engineering: expert parallelism, all-to-all communication, and the **load-balancing problem**. If routing collapses onto a few experts, the remaining parameters are paid for and never trained. The classical remedy is an auxiliary load-balancing loss, which distorts the main objective. DeepSeek-V3's auxiliary-loss-free strategy (a bias-based routing adjustment) and Qwen3's global-batch load-balancing loss are two different resolutions of the same tension. Three questions test any MoE claim: the active parameters per token, the cost of expert-parallel communication on the actual interconnect, and the mechanism that keeps routing balanced.

**MLA is a memory decision.** DeepSeek's multi-head latent attention compresses keys and values into a low-rank latent, which greatly reduces the size of the KV cache. It is presented as an inference feature, but during training it also reduces activation memory and communication, which is part of the reason DeepSeek-V3's GPU-hours are so low. Grouped-query attention (GQA), in which many query heads share fewer key-value heads, is the mainstream version of the same trade. SmolLM3 uses GQA with 4 groups, and Qwen3-8B runs 32 query heads against 8 key-value heads.

**Stability features are now part of the architecture.** QK-norm (RMSNorm applied to queries and keys before the attention dot product) appears in OLMo 2, all Qwen3 models, Gemma 3, and many 2025-2026 reports, because unbounded attention logits are a leading cause of divergence (Ch. 8). OLMo 2 also reordered normalization (it normalizes sublayer outputs) and adopted z-loss on the final softmax. OLMoE measured a throughput cost of almost 10% for QK-norm and kept it, which indicates how much stability is worth. When these features appear in a model card, they usually record a divergence that someone experienced.

**Multi-token prediction (MTP) is a signal-density decision.** DeepSeek-V3 adds a sequential prediction module so that each position also predicts one additional future token (prediction depth 1). This extracts more gradient signal from each sequence, and the module also serves as a speculative-decoding draft at inference. More learning per processed token means fewer GPU-hours per unit of capability.

**Positional-encoding choices are decisions about context cost.** RoPE is universal, and the current refinements aim at inexpensive long-context extension (Ch. 12): partial or adjusted RoPE, hybrid layers without positional encoding (NoPE, used in every fourth layer of SmolLM3), and YaRN-style rescaling at extension time (Kimi K2 to 128K). ALiBi, used in BLOOM, was an earlier attempt at the same goal.

**The general lesson.** Between BLOOM and OPT in 2022 and the frontier models of 2026, the transformer block changed very little. What changed is that every remaining choice is justified by ablations against a cost function that includes training stability and serving cost. An architecture review should be conducted the same way. The question is not whether an idea is elegant, but what it does to tokens per second, memory, spike risk, and the inference bill.

**Checkpoint question.** A model card lists QK-norm, z-loss, and no weight decay on embeddings. What failure does each one address?

*Answer.* QK-norm bounds the attention logits $q^\top k$, whose unbounded growth saturates the softmax and produces pathological gradients. Kimi K2's proxy run saw maximum logits above 1,000 before divergence. Z-loss penalizes growth in the normalizer of the output softmax, which prevents drift in the scale of the final logits. Removing weight decay from embeddings prevents embedding norms from shrinking, which would otherwise inflate early-layer gradients because the LayerNorm Jacobian scales as $1/\lVert x \rVert$. All three come from OLMo 2's stability package and its predecessors (Sec. 8.2).

**References:**
- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
- Kimi K2: Open Agentic Intelligence: https://arxiv.org/abs/2507.20534
- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- 2 OLMo 2 Furious (OLMo 2): https://arxiv.org/abs/2501.00656
- OLMoE: Open Mixture-of-Experts Language Models: https://arxiv.org/abs/2409.02060
- SmolLM3 (Hugging Face blog): https://huggingface.co/blog/smollm3
- YaRN: Efficient Context Window Extension of Large Language Models: https://arxiv.org/abs/2309.00071

## 7. Optimizers, Schedules, and Hyperparameters

*Applies to: frontier labs and regional labs. Sec. 7.3 and Sec. 7.4 also apply to product teams running continued pretraining.*

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

MiniCPM, which introduced WSD, measured how short the decay can be. Loss falls sharply as soon as the learning rate starts to decay, and a decay phase of about 10% of total tokens was enough to reach the best result, while 2.5% fell short. After the decay, WSD matched or outperformed a cosine schedule planned for the same step count. The same paper reports that mixing high-quality and SFT-style data into the decay phase helps far more than introducing the same data only during SFT, which is an early statement of the mid-training idea (Ch. 12). A WSD schedule is a few lines:

```python
import math

def wsd_lr(step, total_steps, peak_lr, warmup_steps=2000, decay_frac=0.10, final_lr=0.0, shape="linear"):
    """Warmup-Stable-Decay learning rate. total_steps can be extended while the run is on the plateau."""
    decay_start = int(total_steps * (1 - decay_frac))
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if step < decay_start:
        return peak_lr
    progress = min(1.0, (step - decay_start) / max(1, total_steps - decay_start))
    if shape == "cosine":
        return final_lr + 0.5 * (peak_lr - final_lr) * (1 + math.cos(math.pi * progress))
    return peak_lr + (final_lr - peak_lr) * progress
```

### 7.4 Hyperparameters that warrant tuning

For a given architecture family, a short list of hyperparameters has real leverage: the **peak learning rate** (the most important setting for both spike risk and final quality), the warmup length, the batch size, the WSD decay fraction, the weight decay, and whether z-loss is enabled. Almost everything else transfers from the reports cited here. The ablation budget should be allocated accordingly.

**Checkpoint question.** Why is a change of batch size in the middle of a run a stability risk?

*Answer.* Adam divides each update by the square root of a running average of squared gradients, and that average was accumulated under the old batch size. After the change the gradient noise is different, so for roughly $1/(1-\beta_2)$ steps the normalizer is miscalibrated. Updates become too large when the stored estimate understates the true second moment, as happens when the batch shrinks. When the batch grows, the normalizer errs in the safe direction, but batch growth is usually paired with a higher learning rate, and the combined change takes effect before the optimizer statistics have adapted. With $\beta_2 = 0.95$ the estimate has a memory of about 20 steps, and with 0.999 about 1,000 steps, which is one reason LLM training uses the smaller value. Batch ramps should therefore be gradual and planned in advance (Sec. 22.2). DeepSeek-V3, for example, raised its batch size from 3,072 to 15,360 sequences gradually over the first 469B tokens.

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

*Applies to: frontier labs and regional labs, and any team running continued pretraining on more than a few billion tokens.*

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

Two of these elements in code. Z-loss is an additional term on the log of the softmax normalizer, and QK-norm is an RMSNorm on the queries and keys:

```python
import torch
import torch.nn.functional as F

def lm_loss_with_z_loss(logits, targets, z_coeff=1e-4, ignore_index=-100):
    """Cross-entropy plus z-loss. logits: [B, T, V], targets: [B, T]."""
    logits = logits.float()                                   # softmax statistics in FP32
    ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=ignore_index)
    log_z = torch.logsumexp(logits, dim=-1)                   # log of the softmax normalizer Z
    mask = (targets != ignore_index).float()
    z_loss = z_coeff * ((log_z ** 2) * mask).sum() / mask.sum().clamp(min=1)
    return ce + z_loss

class QKNormAttention(torch.nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.qkv = torch.nn.Linear(dim, 3 * dim, bias=False)
        self.out = torch.nn.Linear(dim, dim, bias=False)
        self.q_norm = torch.nn.RMSNorm(self.head_dim)         # per head, as in Qwen3;
        self.k_norm = torch.nn.RMSNorm(self.head_dim)         # OLMo 2 normalizes the full projection

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).unbind(dim=2)
        q, k = self.q_norm(q), self.k_norm(k)                 # applied before RoPE
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(B, T, -1))
```

With QK-norm and unit gains, each query and key vector has an RMS of 1, so the logit $q^\top k / \sqrt{d}$ is bounded by $\sqrt{d}$ in magnitude regardless of how the projection weights grow. That bound is the whole mechanism.

### 8.4 Monitoring and leading indicators

The consistent empirical finding is that **the gradient norm leads the loss**. GLM-130B saw collapses lag behind embedding-gradient spikes by a few steps, and OLMo 2's unstable runs show gradient-norm spikes growing in frequency before the loss follows. A run dashboard should show, in order of priority: (1) gradient norms per layer group, with the embedding layer separated; (2) the maximum attention logit per layer, which is essential after Kimi K2's experience; (3) the loss against an exponential moving average of the loss, with an alert threshold; (4) the ratios of parameter norms to update norms; (5) throughput and MFU, where a silent NCCL degradation shows first (Ch. 11); (6) for MoE models, the entropy of the expert load. Alert rules are more reliable than manual intervention. The report for K2-V2 (LLM360's 70B model from MBZUAI, unrelated to Moonshot's Kimi K2) shows a severe spike near step 464,000 being detected automatically, the job rolled back to the last committed checkpoint and relaunched, and a Slack notification sent. That is the operational standard to aim for.

A detector of this kind needs two conditions, severity and persistence, which is how the K2-V2 report describes its monitor. OLMo 2's spike score supplies a reasonable severity threshold: a value at least seven standard deviations from a rolling average over the last 1,000 values.

```python
from collections import deque
import statistics

class SpikeDetector:
    """Returns True when the loss has been far above its recent trend for `patience` consecutive steps."""
    def __init__(self, window=1000, n_sigma=7.0, patience=3, min_history=100):
        self.buf = deque(maxlen=window)
        self.n_sigma, self.patience, self.min_history = n_sigma, patience, min_history
        self.run = 0

    def update(self, loss):
        outlier = False
        if len(self.buf) >= self.min_history:
            mu, sd = statistics.fmean(self.buf), statistics.pstdev(self.buf)
            outlier = loss > mu + self.n_sigma * max(sd, 1e-8)
        self.run = self.run + 1 if outlier else 0
        if not outlier:
            self.buf.append(loss)              # outliers must not contaminate the baseline
        return self.run >= self.patience       # persistent: roll back, relaunch, notify

@torch.no_grad()
def max_attention_logit(q, k):
    """q, k: [B, H, T, D]. Per-head maximum of q.k / sqrt(D). Run it on a small probe batch,
    because it materializes the T x T score matrix that fused attention kernels avoid."""
    scores = q @ k.transpose(-1, -2) / q.size(-1) ** 0.5
    return scores.amax(dim=(0, 2, 3))
```

### 8.5 Restart decision procedure

When a spike occurs despite these measures: if the loss returns to trend within a few hundred steps, log the event and continue (benign). If it does not, rewind to the last healthy checkpoint, skip a window of 200-500 batches around the trigger (the PaLM protocol), optionally reduce the peak learning rate by 10-30%, and resume. If spikes recur at **increasing frequency** (the signature seen in GLM-130B and OLMo), treating symptoms is no longer appropriate. A structural cause (a schedule that is too aggressive, decayed embeddings, logit growth) requires one of the mitigations in Sec. 8.2, and continued rewinding wastes compute. The checkpoint-interval arithmetic that makes this procedure affordable is in Ch. 11.

**Checkpoint question.** A 30B dense run shows gradient-norm spikes in the embedding layer every few thousand steps, and their frequency is rising. The loss has not yet spiked. What should be done?

*Answer.* This is the GLM-130B and OLMo 2 signature of a structural problem, and the gradient norm is a leading indicator, so action is warranted before the loss moves. Check whether weight decay is applied to the embeddings and whether the embedding norm has been shrinking: if so, exempt embeddings from decay (OLMo 2). Check maximum attention logits per layer: if they are growing, add QK-norm or a clipping mechanism. Confirm that $\epsilon$ is 1e-8 and that z-loss is enabled. Changes of this kind are best introduced from a checkpoint before the instability began, and validated on a short forked run, because resuming with a changed architecture (such as added QK-norm) is a larger intervention than changing the optimizer configuration. Rewinding and skipping batches is the wrong tool here, because nothing indicates that a particular batch is responsible.

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

*Applies to: frontier labs and regional labs. Sec. 9.4 applies to every reader who has to fit a model on hardware.*

### 9.1 Precision formats and their failure classes

Each step down in precision roughly doubles arithmetic throughput and halves memory traffic, so every generation of training has moved one format lower, and every move has produced a characteristic class of failures. The history has three periods.

**FP16 (2019-2022): loss scaling.** FP16's narrow exponent range underflows small gradients, so training required *dynamic loss scaling*: the loss is multiplied by a large factor, and the factor is reduced whenever an overflow produces inf or NaN. OPT-175B ran its weights in FP16 with FP32 Adam state and dynamic loss scaling, and the recurring collapse of the loss scale in its logbook is the standard record of how fragile this regime was. GLM-130B's months of spike investigation (Ch. 8) were largely a consequence of its commitment to FP16.

**BF16 (2022 to the present): the stable default.** BF16 keeps FP32's exponent range and gives up mantissa bits. It needs no loss scaling and suffers far fewer range failures, at a small cost in precision that is absorbed by FP32 master weights and FP32 accumulation. Llama 3, OLMo 2, SmolLM3, and nearly every other current model train in BF16. It is the correct default for any reader of this handbook.

**FP8 (2024 to the present): the frontier, with conditions.** DeepSeek-V3 is the first validation of FP8 as the *compute* format at frontier scale, and the report is precise about what made it work. The first element is **fine-grained scaling**: activations are scaled per 1×128 tile and weights per 128×128 block, so that one outlier does not distort the scale of a whole tensor. The second is **promotion of accumulations to higher precision at short intervals**: partial sums are moved to FP32 registers every $N_C = 128$ elements, because accumulation inside the tensor cores alone retained too few bits on their hardware. Sensitive components stay in higher precision. On H800s, FP8 in principle doubles matrix-multiply throughput over BF16, which accounts for a large part of the cost arithmetic in Ch. 2. The general lesson is that **FP8 is not a configuration flag. It is a design for scaling granularity and accumulation policy.** A team that is not prepared to own that design should stay in BF16 and give up nothing except some throughput.

### 9.2 The DeepSeek-V3 FP8 recipe in detail

The formats themselves are fixed by the hardware. The design space is what to quantize, at what granularity, and what to leave alone.

| Format | Exponent / mantissa bits | Largest finite value | Typical role |
|---|---|---|---|
| FP32 | 8 / 23 | about 3.4e38 | Master weights, gradient accumulation, optimizer state |
| FP16 | 5 / 10 | 65,504 | Compute format of the 2019-2022 period, requires loss scaling |
| BF16 | 8 / 7 | about 3.4e38 | Default compute format since 2022 |
| FP8 E4M3 | 4 / 3 | 448 (no infinities) | Weights and activations. DeepSeek-V3 uses it for all tensors |
| FP8 E5M2 | 5 / 2 | 57,344 | Gradients, in the conventional hybrid recipe |
| FP4 E2M1 | 2 / 1 | 6 | Element format of MXFP4 and NVFP4, always with block scales |

The conventional FP8 recipe uses E4M3 in the forward pass and E5M2 for gradients, because gradients need range more than precision. DeepSeek-V3 uses **E4M3 for all tensors**, and the report credits fine-grained scaling for making that possible: with a scale per 128 values, computed online from the maximum absolute value of each tile or block, the dynamic range within a group is small enough for E4M3. The remaining choices are as follows.

- **What runs in FP8.** All three matrix multiplications of each linear layer: the forward pass, the activation gradient, and the weight gradient.
- **What stays in BF16 or FP32.** The embedding module, the output head, the MoE gating modules, the normalization operators, and the attention operators.
- **Optimizer state.** The AdamW first and second moments are held in BF16 instead of FP32, while the master weights and the gradients used for accumulation across micro-batches stay in FP32.
- **Accumulation precision.** FP8 matrix multiplication on the H800 retains about 14 bits in its internal accumulation, which for an inner dimension of 4,096 produced a maximum relative error of nearly 2% in their test. Promoting partial sums to FP32 every 128 elements removes most of the error at small cost.
- **Validation.** Against a BF16 baseline, the relative loss error stayed below 0.25% at two model scales trained for about 1T tokens.

### 9.3 Block-scaled formats: MXFP8 and NVFP4

Blackwell-generation GPUs implement block scaling in hardware, which turns DeepSeek-V3's software technique into a standard format. Two NVIDIA reports define the current recipes.

**MXFP8.** Blocks of 32 elements share a power-of-two scale (UE8M0). In NVIDIA's MXFP8 recipe paper, an 8B dense model trained on 15T tokens kept its validation perplexity within 0.50% of the BF16 run throughout, with matching downstream scores. The recipe uses E4M3 for weights, activations, *and* gradients (the 8B model degraded with E5M2 gradients), quantizes all linear layers inside the transformer blocks, and leaves embeddings, the output projection, attention matrix products, softmax, and normalization in BF16. One detail matters for stability: the scale exponent must be rounded *up*. The rounding-down suggested in the first version of the OCP specification causes saturation and can lead to divergence.

**NVFP4.** Elements are 4-bit E2M1 values in blocks of 16, each block with an E4M3 scale, plus an FP32 scale per tensor. NVIDIA trained a 12B hybrid Mamba-Transformer on 10T tokens in NVFP4. Relative to the FP8 baseline, the loss gap stayed below 1% during the stable phase and widened to slightly above 1.5% during learning-rate decay, and downstream accuracy was close (MMLU-Pro 62.58 against 62.62). The recipe keeps about 15% of the linear layers in BF16 (the first two blocks and the last eight), applies random Hadamard transforms to the inputs of the weight-gradient multiplication, uses stochastic rounding on gradients only, and keeps master weights, weight gradients, and optimizer state in FP32. For comparison, MXFP4 needed 36% more tokens to reach the same loss in an 8B experiment.

The practical reading in 2026: BF16 remains the default, FP8 with block scaling is production-ready on hardware that supports it natively, and FP4 training is demonstrated but still requires a mixed-precision layout and close monitoring during the decay phase.

### 9.4 Training memory accounting

Per parameter, mixed-precision AdamW training costs roughly 2 bytes for weights (BF16), 2-4 bytes for gradients, 8 bytes for the Adam moments (FP32 $m$ and $v$), and 4 bytes for the FP32 master weights: **about 16-18 bytes per parameter, before activations**. A 70B model therefore carries about 1.2TB of state. This is the reason for sharding (ZeRO and FSDP, Ch. 10) and for activation checkpointing (recomputing activations in the backward pass instead of storing them). BLOOM provides an empirical check: a full 176B checkpoint with optimizer state is 2.3TB, about 13 bytes per parameter on disk, while the BF16 weights alone are 329GB. Storage and restart times should be planned from these figures.

**Worked example.** An 8B model with BF16 weights, FP32 gradients accumulated across micro-batches, and FP32 Adam state carries $8\text{e}9 \times 16 = 128$ GB of model state. No 80GB GPU holds it. Fully sharded across 8 GPUs (ZeRO stage 3 or FSDP), each GPU holds 16 GB of state and gathers one layer's parameters at a time.

Activations are the larger problem. The standard estimate from Korthikanti et al. for one transformer layer with 16-bit activations is $sbh\,(34 + 5as/h)$ bytes, where $s$ is the sequence length, $b$ the micro-batch size, $h$ the hidden size, and $a$ the number of attention heads. For a model shaped like Llama 3 8B ($h = 4096$, $a = 32$, 32 layers) at $s = 8192$ and $b = 1$:

- Stored naively, the attention term dominates: $5as/h = 320$, so one layer needs $8192 \times 4096 \times 354 \approx 11.9$ GB, and 32 layers need about 380 GB.
- With FlashAttention the $s \times s$ attention matrices are never stored, so the second term disappears: about 1.14 GB per layer, 36.5 GB in total.
- With full activation checkpointing only each layer's input is kept ($2sbh$ bytes, 67 MB per layer, about 2.1 GB in total), and one layer's activations are recomputed at a time during the backward pass, at the cost of roughly one additional forward pass.

The constants were derived for a GPT-style block with a 4h MLP and dropout, so a SwiGLU block with grouped-query attention differs by a modest factor, but the orders of magnitude hold. They explain why FlashAttention and some form of recomputation are universal (Sec. 10.4).

**Checkpoint question.** A team wants to fine-tune a 70B model with full parameters on 8 GPUs with 80GB each. Does the model state fit?

*Answer.* No. At 16 bytes per parameter the state is about 1.12 TB, and 8 × 80 GB is 640 GB before any activations. The options are: more GPUs (at least 16 to 24, with full sharding and activation checkpointing); an 8-bit or paged optimizer, which reduces the 8 bytes of Adam state; or a parameter-efficient method. With LoRA the 70B base weights are frozen at 2 bytes per parameter (140 GB, sharded across the GPUs), and gradients and optimizer state exist only for the adapter, which is well under 1% of the parameters. With QLoRA the frozen base is quantized to 4 bits, about 35-40 GB (Ch. 19).

**References:**
- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
- OPT: Open Pre-trained Transformer Language Models: https://arxiv.org/abs/2205.01068
- GLM-130B: https://arxiv.org/abs/2210.02414
- BLOOM model card (checkpoint sizes): https://huggingface.co/bigscience/bloom
- FP8 Formats for Deep Learning: https://arxiv.org/abs/2209.05433
- Recipes for Pre-training LLMs with MXFP8: https://arxiv.org/abs/2506.08027
- Pretraining Large Language Models with NVFP4: https://arxiv.org/abs/2509.25149
- Reducing Activation Recomputation in Large Transformer Models (Korthikanti et al.): https://arxiv.org/abs/2205.05198

## 10. Distributed Training: Parallelism Strategies and Measured MFU

*Applies to: frontier labs and regional labs. Sec. 10.4 applies to anyone training on even a single GPU.*

### 10.1 Parallelism dimensions

Training parallelism comes down to what is split: the batch, the optimizer and parameter state, individual weight matrices, the layer stack, or the experts.

- **Data parallelism (DP):** replicate the model, split the batch, and all-reduce the gradients. It scales easily and is memory-hungry until the state is sharded.
- **ZeRO and FSDP:** still data parallelism, but the optimizer state (stage 1), the gradients (stage 2), and the parameters (stage 3) are *sharded* across ranks and gathered just in time. This is how OPT ran (FSDP with the FP32 Adam state sharded across hosts) and how most training below 100B parameters runs today.
- **Tensor parallelism (TP):** split individual weight matrices across GPUs. It requires communication inside every layer, so it is placed where bandwidth is highest. SmolLM3 used TP=2 strictly within a node over NVLink, with data parallelism across nodes over EFA, because inter-node bandwidth is an order of magnitude lower. **The bandwidth hierarchy dictates the parallelism layout, and not the reverse.**
- **Pipeline parallelism (PP):** split the layers into stages. The cost is idle time ("bubbles") at batch boundaries, which is managed with micro-batching and scheduling. DeepSeek's DualPipe redesigned the schedule to overlap forward and backward computation with communication, because the restricted interconnect of the H800 made hiding communication essential.
- **Expert parallelism (EP):** the dimension specific to MoE. Experts are distributed across GPUs, and tokens are routed by all-to-all communication. It combines with all of the above and is the dominant communication cost in large MoE training.

Large runs compose these dimensions. Llama 3.1 405B ran 4D parallelism (TP × PP × context parallelism × DP) across 16,384 GPUs.

The layouts of the documented runs show how differently the dimensions are combined:

| Run | Layout | Note |
|---|---|---|
| BLOOM-176B (384 A100) | TP=4, PP=12, DP=8, ZeRO-1 | Megatron-DeepSpeed, from the BigScience training README |
| Llama 3.1 405B (16,384 H100) | TP=8, PP=16, DP=128 at 8K sequence length. For the 131K stage: TP=8, CP=16, PP=16, DP=8 | 43% MFU at 8,192 GPUs, 41% at 16,384, 38% with context parallelism |
| DeepSeek-V3 (2,048 H800) | PP=16 (DualPipe), EP=64 across 8 nodes, ZeRO-1 DP, no TP | Tensor parallelism avoided because of its communication cost |
| Kimi K2 (H800) | PP=16 with virtual stages, EP=16, ZeRO-1 DP, no TP | Runs on any node count that is a multiple of 32 |
| SmolLM3 3B (384 H100) | TP=2 within the node, DP=192 across nodes | nanotron |

Two patterns stand out. The large MoE runs avoid tensor parallelism entirely and spend their communication budget on expert parallelism. Context parallelism appears only when sequences become long: Llama 3 implements it by all-gathering keys and values and computing attention for the local query chunks, a choice the paper justifies by the small size of K and V under GQA and by the ease of supporting document masks.

### 10.2 MFU in production: the MegaScale reference

ByteDance's MegaScale (NSDI 2024) is the best public account of raising MFU in production: **55.2% on a 175B model across 12,288 GPUs**, a 1.34× improvement over Megatron-LM. It was achieved through full-stack co-design (overlapped communication, fused operators, data-pipeline optimization, network tuning) and through deep observability, discussed in Sec. 10.3, because at that scale a single slow GPU silently limits the whole job. The figures also calibrate expectations. A major production effort reaches about 55%. A good run by a small team (SmolLM3) reaches about 30% end to end, even with matrix-multiply kernels at 72-77% of peak. A first multi-node run at 25% is normal, and the gap decomposes into exposed communication (overlap it), input-pipeline stalls (prefetch and pre-tokenize), kernel inefficiency (fused attention and MLP kernels), and stragglers.

### 10.3 Storage, checkpointing, and observability

- **Checkpointing arithmetic.** The time to write a checkpoint scales with the state size (Sec. 9.4) divided by the storage bandwidth. If checkpointing stalls training, the choices are to checkpoint less often (which raises the expected work lost per failure, Ch. 11) or to fix the I/O path. Storage can also throttle the data path: during SmolLM3's run, throughput collapsed because the shared network filesystem (Weka, backed by S3) began evicting shards of the 24TB training dataset. The fix was to copy the dataset onto each node's local NVMe RAID and to keep a spare node preloaded with it, so that replacing a failed node cost no download time. The pattern to copy is training data and hot checkpoints on node-local NVMe, durable copies in object storage, and no shared filesystem in the critical path.
- **Data loading is a correctness surface, not only a throughput one.** The single most expensive defect of the SmolLM3 project was in the parallelism code: **all tensor-parallel ranks were initialized with the same random seed**, which silently degraded learning. The model underperformed its smaller predecessor at the same stage of training, and after the investigation the team **restarted the run, having spent 1T tokens**. Determinism tests (the same seed gives the same loss for 100 steps under every parallelism configuration, and seeds that must differ are verified per rank) are far cheaper than that.
- **Observability.** MegaScale instruments the stack deeply (per-rank timing, tracing of collective communication, hardware counters), because at more than 10,000 GPUs a complaint that training "feels slow" has ten thousand possible causes. Even at 8 GPUs, per-rank step timings should be kept. The first straggler is often one faulty DIMM or a thermally throttled card, and averaged metrics hide it.

### 10.4 Single-GPU throughput techniques

Most of the distance between a naive implementation and a published MFU figure is covered by four techniques that apply at every scale.

**FlashAttention.** Standard attention materializes an $s \times s$ score matrix per head, which makes memory quadratic in sequence length and makes the operation bound by memory traffic. FlashAttention computes exact attention in tiles that fit in on-chip SRAM, needs only $O(s)$ additional memory, and recomputes the tiles in the backward pass from saved softmax statistics. FlashAttention-2 roughly doubled the speed of the original and reached 225 TFLOP/s per A100 in end-to-end GPT training, which is 72% MFU. FlashAttention-3 targets the H100: 1.5 to 2 times faster than FlashAttention-2, up to 740 TFLOP/s in FP16 (75% utilization), with an FP8 path. A fourth generation for Hopper and Blackwell GPUs was in beta at the time of writing. For training, FlashAttention is a default, not an optimization.

**Activation recomputation.** Storing every activation for the backward pass is usually impossible (Sec. 9.4). Full recomputation stores only layer inputs and costs one additional forward pass, which Korthikanti et al. measured at 30-40% of execution time. *Selective* recomputation discards only the attention-score tensors, which hold about 70% of activation memory in a GPT-3-sized layer and cost only 2.7% of its FLOPs to recompute. Combined with sequence parallelism it reduced activation memory by a factor of five and recovered more than 90% of the recomputation overhead, reaching 54.2% MFU on a 530B model across 2,240 A100s against 42.1% with full recomputation. With FlashAttention the attention scores are never stored, so the remaining decision is usually how many layers to checkpoint.

**Sequence packing with document masking.** Pretraining documents are concatenated into fixed-length sequences so that no compute is spent on padding. Llama 3 adds an attention mask that prevents tokens from attending across document boundaries within a sequence. The paper reports limited impact in standard pretraining and an important effect in continued pretraining on very long sequences. SmolLM3 adopted the same intra-document masking. In SFT, where examples are short and variable in length, packing is the difference between a GPU that is mostly computing and one that is mostly padding, and it requires the same masking and correct position resets (Ch. 14).

**Fused kernels and compilation.** Normalization, rotary embeddings, SwiGLU, and above all the final cross-entropy over a large vocabulary are dominated by memory traffic, not arithmetic. Fused implementations, such as the Triton kernels in LinkedIn's Liger Kernel, report about 20% higher throughput and about 60% lower memory than the reference Hugging Face implementations, largely by never materializing the full logits tensor. `torch.compile` obtains part of the same benefit automatically.

**Worked example: computing MFU.** The Smol Training Playbook's 1B ablation model, which follows the Llama 3.2 1B architecture (1.23B parameters), trains at about 42,000 tokens per second per H100. Each token costs about $6N$ FLOPs, so one GPU delivers $6 \times 1.23\text{e}9 \times 42\text{e}3 \approx 3.1\text{e}14$ FLOP/s, which is 31% of the H100's 989 TFLOP/s. The $6N$ rule ignores attention FLOPs, which are small at a 4K sequence length and become significant at long context, so the figure slightly understates the true utilization.

**Checkpoint question.** A 3B model on 64 H100s reports 18% MFU. In what order should the causes be investigated?

*Answer.* (1) Per-rank step times, to find a straggler or a throttled GPU, because synchronous training runs at the speed of the slowest rank. (2) The input pipeline: if GPU utilization drops between steps, the data loader or the storage is the limit. (3) Exposed communication: a profile showing long all-reduce or all-gather segments that do not overlap with computation points to the parallelism layout, for example tensor parallelism crossing node boundaries. (4) Kernels: confirm FlashAttention, fused cross-entropy, and BF16 matrix multiplies are in use, and that the micro-batch is large enough to saturate the GPU. (5) Recomputation: full activation checkpointing alone costs up to a third of throughput, and at 3B on 80GB GPUs selective recomputation or none may fit. SmolLM3, a 3B model on 384 H100s, reached about 30%, which is the realistic target.

**References:**
- MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs: https://arxiv.org/abs/2402.15627
- The Llama 3 Herd of Models: https://arxiv.org/abs/2407.21783
- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
- OPT: Open Pre-trained Transformer Language Models: https://arxiv.org/abs/2205.01068
- ZeRO: Memory Optimizations Toward Training Trillion Parameter Models: https://arxiv.org/abs/1910.02054
- Megatron-LM: https://arxiv.org/abs/1909.08053
- The Smol Training Playbook (Hugging Face): https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook
- SmolLM3 (Hugging Face blog): https://huggingface.co/blog/smollm3
- BLOOM training README (BigScience): https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/README.md
- Kimi K2: Open Agentic Intelligence: https://arxiv.org/abs/2507.20534
- FlashAttention: https://arxiv.org/abs/2205.14135 · FlashAttention-2: https://arxiv.org/abs/2307.08691 · FlashAttention-3: https://arxiv.org/abs/2407.08608
- Reducing Activation Recomputation in Large Transformer Models (Korthikanti et al.): https://arxiv.org/abs/2205.05198
- Liger Kernel: https://arxiv.org/abs/2410.10989

## 11. Hardware Reliability at Scale

*Applies to: frontier labs and regional labs, and any team whose runs last longer than a day.*

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

**Worked example: what the checkpoint write time is worth.** At the optimum, the fraction of time lost to checkpointing and rework is $\sqrt{2\,t_{write}/M}$. With $M = 3$ h and a blocking 5-minute write, the optimal interval is $\sqrt{2 \times 180 \times 5} \approx 42$ minutes, and the overhead is $\sqrt{10/180} \approx 24\%$, which is incompatible with Llama 3's reported effective training time above 90%. If the training stall per checkpoint is cut to 30 seconds, by writing to local NVMe or host memory and uploading asynchronously, the optimal interval falls to about 13 minutes and the overhead to about 7.5%, before restart time is counted. The same logic explains a design choice in the Gemini report: keeping redundant in-memory copies of the model state and recovering from an intact replica, instead of relying on periodic checkpoints to persistent storage, raised goodput for its largest job from 85% to 97%.

```python
import math

def checkpoint_plan(mtbf_minutes, stall_minutes):
    """Young-Daly first-order optimum. `stall_minutes` is how long training is blocked per checkpoint."""
    interval = math.sqrt(2 * mtbf_minutes * stall_minutes)
    overhead = stall_minutes / interval + interval / (2 * mtbf_minutes)
    return interval, overhead

checkpoint_plan(180, 5.0)   # (42.4 minutes, 0.236)
checkpoint_plan(180, 0.5)   # (13.4 minutes, 0.075)
```

**Checkpoint question.** A 2,048-GPU job runs on nodes whose failure rate is 6.5 failures per 1,000 node-days. What mean time between failures should the team plan for, and how often should it checkpoint if a checkpoint blocks training for one minute?

*Answer.* 2,048 GPUs are 256 nodes of 8. The failure rate is $256 \times 6.5 / 1000 = 1.66$ failures per day, so the mean time between failures is about 14.4 hours, or 865 minutes. This is the first-order model and the rate that Meta's reliability study fitted to its RSC-1 cluster. The Young-Daly interval is $\sqrt{2 \times 865 \times 1} \approx 42$ minutes, and the expected overhead is about 4.8%, plus the time to restart after each failure.

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

*Applies to: frontier labs and regional labs. The closing section applies to product teams planning continued pretraining.*

Mid-training is the point at which pretraining changed from a single undifferentiated mixture of tokens to a staged curriculum. It is the highest-return idea that a small team can adopt from frontier reports, because it costs a fraction of pretraining and improves benchmarks disproportionately.

**The pattern across laboratories.** OLMo 2 trains a first stage on a broad, web-dominated mixture, and then restarts from that checkpoint on domain-specific mixtures (mathematics-heavy and curated high-quality data) while driving the learning rate linearly to zero. Its microannealing runs are small versions of the same procedure, used to score candidate data (Sec. 3.2). Qwen3 uses three explicit stages: more than 30T general tokens, about 5T tokens dense in STEM, code, and reasoning, and then long-context data at 32K. Kimi K2 ends its 15.5T-token run with a 400B-token annealing phase (learning rate 2e-5 to 7e-6) followed by a 60B-token stage at a sequence length of 32K. SmolLM3 runs three stages, with the mixture upgraded as the WSD decay begins. The shared logic is that **the decay phase is when the model consolidates, so it should receive the best tokens.** High-quality data used early is partly overwritten. Used late, it persists.

### 12.1 Documented recipes

| Model | Mid-training stage | Detail |
|---|---|---|
| Llama 3 405B | Annealing over the final 40M tokens | Learning rate annealed linearly to zero at 128K context, high-quality sources upsampled, final model is the average of checkpoints taken during annealing |
| OLMo 2 (7B, 13B, 32B) | 50B, 100B, or 300B tokens sampled from the Dolmino mix (a pool of about 843B tokens) | Learning rate decayed linearly to zero. 7B: three 50B-token runs with different data orders, averaged. 13B and 32B: three 100B-token runs and one 300B-token run, averaged |
| SmolLM3 3B | Stage 2 (8T to 10T tokens) and stage 3 (10T to 11.1T, the decay stage) | Web / code / mathematics: 85 / 12 / 3 in stage 1, 75 / 15 / 10 in stage 2, 63 / 24 / 13 in stage 3, with better mathematics and code sources and instruction-style reasoning data added in the last stage |
| Qwen3 | About 5T tokens (stage 2), then a long-context stage | Stage 2 raises the share of STEM, code, and reasoning data at a 4K sequence length |
| Kimi K2 | 400B tokens at 4K, then 60B at 32K | Learning rate decayed from 2e-5 to 7e-6 across the phase, batch size constant at 67M tokens |
| MiniCPM | The WSD decay phase | High-quality and SFT-style data mixed into the decay phase outperformed the same data used only in SFT |

Llama 3 also reports a useful negative result. Annealing on the GSM8K and MATH *training sets* raised an 8B model's scores by 24.0% and 6.4% on the corresponding validation sets, but the gains on the 405B model were negligible, which the authors attribute to the larger model's in-context learning and reasoning ability. Small models benefit most from targeted annealing data. The same paper turns annealing into a measurement tool: a 50%-trained 8B model is annealed over 40B tokens on a mixture of 30% candidate data and 70% default mix, which the authors found more efficient than running scaling-law experiments for every small dataset.

### 12.2 Checkpoint averaging

Averaging weights across checkpoints is now a standard part of the stage. Llama 3 averages checkpoints within the annealing run. OLMo 2 averages the results of *separate* annealing runs that differ only in data order, and reports that the merged model is equal to or better than any single run. SmolLM3 extends the idea to post-training, where a merge repaired a regression: after preference optimization its long-context scores (RULER) fell, and linearly merging the averaged preference-optimized checkpoints with a mid-training checkpoint at weights 0.9 and 0.1 recovered the base model's long-context performance up to 128K. Averaging is inexpensive, and it should be tested wherever a stage ends with several comparable checkpoints.

### 12.3 Long-context extension

**Long-context extension is inexpensive and late.** The recurring recipe is to train almost everything at a short context (4K), then run a brief stage on naturally long documents at 32K, and then extrapolate the RoPE geometry (YaRN and related methods) to the advertised window (Kimi K2: 60B tokens at 32K, then YaRN to 128K). Two cautions emerge from the collective experience. Long-context data must consist of *naturally* long documents, not concatenated short ones, or the model learns to ignore distant context. Every advertised window should also be verified with retrieval-style evaluations at full depth, because extension methods degrade gradually and without obvious symptoms.

The documented procedures:

| Model | Procedure |
|---|---|
| Llama 3 405B | Six stages from 8K to 128K, about 800B tokens. The next stage begins only when short-context evaluations have recovered and needle-in-a-haystack retrieval is solved at the current length |
| DeepSeek-V3 | Two YaRN phases of 1,000 steps each: 4K to 32K (batch size 1,920), then 32K to 128K (batch size 480), both at the final pretraining learning rate of 7.3e-6. Total cost: 119K GPU-hours |
| Qwen3 | Hundreds of billions of tokens at 32,768. 75% of the text is 16,384 to 32,768 tokens long and 25% is 4,096 to 16,384. RoPE base frequency raised from 10,000 to 1,000,000 (ABF). YaRN and Dual Chunk Attention give a further fourfold extension at inference |
| SmolLM3 | 100B additional tokens in two 50B stages: 4K to 32K with RoPE theta 1.5M, then 32K to 64K with theta 5M. YaRN extends 64K to 128K at inference. Upsampling long documents gave no further gain beyond NoPE layers and the larger theta |
| Kimi K2 | 60B tokens at 32K after the 4K annealing phase, then YaRN to 128K |

The cost is consistently small. DeepSeek-V3's extension used 119K of 2.788M GPU-hours, about 4%. Llama 3's 800B tokens are about 5% of its 15.6T. The gating rule in Llama 3's procedure is the part most worth copying: advance only when short-context quality has recovered and retrieval at the current length is solved.

### 12.4 Relevance to smaller teams

**Relevance to smaller teams.** For a team that holds an open base model and a modest cluster, continued pretraining in the style of mid-training (a few tens of billions of curated domain tokens, trained through a decay schedule) is frequently the best available improvement per dollar. It is also the mechanism behind language adaptation, the subject of the next chapter.

**Checkpoint question.** A team has 20B tokens of high-quality Arabic educational text and 300B tokens of general Arabic web text for a continued-pretraining run. Where in the run should the high-quality tokens be placed, and why?

*Answer.* Concentrated in the final phase, while the learning rate decays. MiniCPM's experiments show high-quality data in the decay phase outperforming the same data used only afterwards, and every staged recipe above (OLMo 2, Qwen3, SmolLM3, Kimi K2, Llama 3) upgrades the mixture as the rate falls. Data seen at a high learning rate is partly overwritten by later updates, while data seen during the decay determines where the model settles. A small share can appear earlier so that the distribution is not new when the decay begins, and original-language replay should continue throughout (Sec. 13.2). If several decay runs are affordable, run them with different data orders and average the results.

**References:**
- 2 OLMo 2 Furious (OLMo 2): https://arxiv.org/abs/2501.00656
- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- Kimi K2: Open Agentic Intelligence: https://arxiv.org/abs/2507.20534
- SmolLM3 (Hugging Face blog): https://huggingface.co/blog/smollm3
- YaRN: Efficient Context Window Extension of Large Language Models: https://arxiv.org/abs/2309.00071
- The Llama 3 Herd of Models: https://arxiv.org/abs/2407.21783
- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
- MiniCPM: https://arxiv.org/abs/2404.06395

## 13. Continued Pretraining and New-Language Adaptation: The Arabic Case

*Applies to: regional labs and product teams. Frontier labs and national programs should read Sec. 13.1 on the from-scratch path.*

This chapter is the center of the handbook for readers in the Middle East and North Africa. It sets out the decision space for making a model good at Arabic, or at any underserved language, reconstructed from the published record of Jais, AceGPT, ALLaM, Kuwain, and RightNow-Arabic and from the general literature on continued pretraining. The same procedure applies to domain adaptation (legal, medical, dialectal speech transcripts). Arabic supplies the most demanding and best-documented case.

### 13.1 The three strategic paths

**Path A: from scratch, bilingual by design (Jais, 2023, and ALLaM's from-scratch model).** Jais trained a 13B GPT-3-style decoder from random initialization on 395B tokens, with Arabic deliberately given a large share: 72B unique Arabic tokens (the largest Arabic corpus assembled at the time), upsampled to about 116B (roughly 1.6 epochs), against English and code at an Arabic-to-English ratio of roughly 1:2, with a purpose-built balanced bilingual tokenizer. The premise, which the results supported, was cross-lingual transfer: abundant English supplies world knowledge and reasoning while Arabic anchors the linguistic space. The model outperformed existing open models on Arabic by a wide margin while remaining competitive in English. The cost profile is the highest of the three paths: full pretraining prices (Ch. 2) and exposure to every problem in this handbook. The follow-up Jais family (2024) scaled the approach to 20 models from 590M to 70B parameters, trained on up to 1.6T tokens, and released *both* from-scratch and Llama-2-adapted variants, which indicates that Path B had become competitive.

**Path B: continued pretraining of a strong open base (AceGPT, 2023, the Llama-2-based ALLaM models, and the default choice in 2026).** AceGPT continued Llama-2 (7B and 13B) on Arabic-majority mixtures (60-64% Arabic) of 30B and 10B tokens respectively, followed by localized post-training (Arabic instructions, and RLAIF with a reward model tuned to local culture and values). This obtains most of the benefit for a fraction of the from-scratch cost. ALLaM's team ran the most informative comparison. They first adapted Llama-2 through tokenizer and vocabulary expansion and 1.2T tokens of mixed Arabic-English continued pretraining (600B for the 70B model), *then* applied the recipe from scratch, and reported that the from-scratch 7B improved significantly over the continued 7B. The comparison is reported briefly and is not compute-matched: the from-scratch model saw 4T English tokens followed by the same 1.2T mixed tokens, against Llama-2's 2T plus 1.2T. A careful reading is that adaptation delivers most of the result quickly and cheaply, and that training from scratch obtains the remainder at a price justified mainly at national-program budgets. For other organizations, Path B on a 2026-class base (Qwen3, Llama, Gemma) starts from a far stronger prior than Llama-2 offered.

The Jais family's adapted models document the budgets involved. According to the model card, the Llama-2-based variants added 32K Arabic tokens from the Jais-30B vocabulary, initialized the new embeddings through a learned projection from the Jais-30B embedding space, trained only the embeddings for a first stage of about 15B tokens, and then continued pretraining on 39B tokens (7B), 280B tokens (13B), and 371B tokens (70B, of which 334B Arabic). Gosal et al. is the closest published description of this recipe. An embedding-only first stage is an inexpensive precaution that protects the backbone while the new rows are still far from converged.

**Path C: targeted adaptation of a small model (Kuwain, 2025, and RightNow-Arabic-0.5B, 2026).** This is the budget tier and the most instructive one for teams with a single node. Kuwain-1.5B extended TinyLlama-1.1B in two ways at once: 26K new Arabic tokens in the tokenizer, and 8 new decoder layers trained on 110B tokens (90B Arabic, 20B English) with the original layers frozen, except the last, which was kept trainable for stability. The Arabic benchmark average rose from 36.95 to 44.49, and English held (52.99 to 53.28). The frozen backbone matters: the paper's own ablation with vocabulary expansion and ordinary continued pretraining, without the new layers, reached comparable Arabic scores but reduced the English average to 46.85. RightNow-Arabic-0.5B-Turbo documents a complete small-scale recipe: add 27,032 Arabic tokens to Qwen2.5-0.5B, continue pretraining on about 504M Arabic tokens, run SFT with response-only loss masking, run DPO, merge the pretrain, SFT, and DPO checkpoints linearly (25/25/50), and export to GGUF for edge devices (398MB at 4 bits, 635 tokens per second on one H100 at batch size 1). Total compute was one 8×H100 node for under eight hours. The reported gains are correspondingly modest: a mean accuracy of 35.9 against 34.1 for Qwen2.5-0.5B-Instruct, with COPA-ar and HellaSwag-ar up 4.5 and 3.5 points and ArabicMMLU down 2.8. This tier exists because of what Paths A and B established.

### 13.2 The vocabulary-expansion operation, step by step

The recurring mechanical core of Paths B and C, assembled from ALLaM, Kuwain, and RightNow-Arabic:

1. **Train an Arabic tokenizer, or merge a bilingual one,** and select the new tokens to add. More tokens give better fertility and a larger embedding matrix. At 0.5B scale RightNow-Arabic chose about 27K tokens, while ALLaM merged a full Arabic tokenizer into Llama-2's, which brought Arabic fertility down to the level of an Arabic-only tokenizer. For scale, Gosal et al. measured Llama-2's Arabic fertility falling from 5.06 to 1.41 tokens per word after adding 32K Arabic tokens (Ch. 4), a 72% reduction in the token cost of every subsequent Arabic training and inference step.
2. **Extend the embedding and output matrices,** initializing each new token's vectors from the mean of the embeddings of its decomposition under the old tokenizer: tokenize the new token's string with the *old* tokenizer and average those rows. This is subtoken-mean initialization (Fast Vocabulary Transfer, Gee et al., 2022), which ALLaM and RightNow-Arabic both use. Kuwain's paper does not state how its new embeddings were initialized. ALLaM reports much faster learning with this method than with random initialization. Random initialization slows adaptation because it forces the model to relearn, under new names, what it already knows.
3. **Continue pretraining on a mixed corpus, never a pure one.** Catastrophic forgetting is real, and the remedy is *anchoring*. ALLaM argues explicitly that vocabulary expansion combined with a continued English presence in the mixture is what prevents forgetting of the base model's abilities. The recorded ratios run from an Arabic majority with an English anchor (AceGPT) to blends near 1:1, and ALLaM reports that 45% Arabic to 55% English worked best in its setting. The replay ratio should be treated as an ablation (Sec. 5.3), with 10-30% original-distribution data as the common starting range in the broader literature on continued pretraining. Replay alone may not be enough at small scale: in Kuwain's ablation, ordinary continued pretraining after vocabulary expansion lost about six points of English average (52.99 to 46.85), while training only inserted layers over a frozen backbone held English steady.
4. **Treat the learning rate as an open question.** The published evidence conflicts. ALLaM ran all continued pretraining at the base model's final learning rate (3e-5 for Llama-2) and reports that schedules which re-warmed and then decayed the rate had limited success and typically caused forgetting of English. Gosal et al., adapting the same Llama-2 base, found the opposite: re-warming to the full peak of 3e-4 with 1% linear warmup and cosine decay worked best, on a 9:1 Arabic-to-English mixture. The two studies differ in mixture and token budget, so neither result transfers without testing. Re-warming perturbs a converged optimizer state (the mechanisms of Ch. 8 apply), so the prudent procedure is to start low and raise the rate only when an ablation shows a gain.
5. **Evaluate in both directions.** Evaluate Arabic *and* the original languages every few billion tokens. Forgetting appears first in the tails (code, mathematical word problems) before headline English benchmarks move.

Step 2 in code, for a Hugging Face model. The subtoken decomposition must be computed *before* the tokens are added:

```python
import torch

@torch.no_grad()
def add_tokens_with_subtoken_mean(model, tokenizer, candidates):
    """Add tokens and initialize each new embedding row as the mean of the rows of the pieces
    that the ORIGINAL tokenizer produces for the token's string (subtoken-mean initialization)."""
    vocab = tokenizer.get_vocab()
    new_tokens = [t for t in dict.fromkeys(candidates) if t not in vocab]
    pieces = [tokenizer.encode(t, add_special_tokens=False) for t in new_tokens]
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))
    emb_in = model.get_input_embeddings().weight
    emb_out = model.get_output_embeddings().weight
    for token, ids in zip(new_tokens, pieces):
        new_id = tokenizer.convert_tokens_to_ids(token)
        emb_in[new_id] = emb_in[ids].mean(dim=0)
        if emb_out.data_ptr() != emb_in.data_ptr():      # untied output head
            emb_out[new_id] = emb_out[ids].mean(dim=0)
    return new_tokens
```

Production pipelines merge a trained target-language BPE vocabulary, tokens and merge rules together, into the tokenizer instead of registering added tokens, but the embedding initialization is identical. Two checks should follow. The loss on Arabic text immediately after the operation should be close to the loss before it, because each new token starts as an average of what it replaces. The loss on English text should be unchanged.

### 13.3 Arabic data supply and translated data

ALLaM's Arabic corpus is 540B tokens, and only half of it is natural: 270B tokens of curated Arabic, a large engineering effort given the state of the public Arabic web (Sec. 3.3), and 270B tokens machine-translated from English sources with an in-house translation system. The choice is controversial but supported by the paper's ablations, which show translated data reducing gradient spikes and helping to align Arabic and English capability. Jais's Arabic likewise included a substantial translated share. The pragmatic reading is that translation artifacts (unnatural style, calqued idioms) are a real cost that is accepted knowingly, because token supply is the binding constraint. The mitigation is to concentrate natural, native Arabic, including dialects, in the *late* high-influence stages (Ch. 12) and in post-training, where style is set. For dialects specifically (the gap between MSA benchmarks and the way Gulf users type), the corpus problem is harder, and the current answer is commissioned and annotated data, not scraping. That is a data-operations problem, not a modeling one.

### 13.4 Decision framework

| Situation | Recommended path | Reference recipe |
|---|---|---|
| National program with a nine-figure budget and sovereignty requirements | A (from scratch), after running B first as a de-risking study | ALLaM's sequence: adapt, learn, then train from scratch |
| Company with a substantial cluster (hundreds of GPUs) and an Arabic product | B: vocabulary expansion, tens to hundreds of billions of mixed tokens on a 2026 open base, and full post-training | AceGPT and the Llama-2-based ALLaM, on a current base |
| Team with one or two nodes | C: token injection (optionally with inserted layers over a frozen backbone), continued pretraining sized to the budget (0.5B tokens in RightNow-Arabic, 110B in Kuwain), and SFT with DPO | Kuwain, RightNow-Arabic |
| Application team with no mandate to train | None of the above: use the existing ecosystem (the Jais family, ALLaM releases, SILMA, Fanar, Qwen3's multilingual models) and spend the budget on evaluation and post-training data | Ch. 14-19 |

The error to avoid at every tier is judging success by MSA benchmarks alone. The published models cluster together on translated MMLU-style evaluations. They *diverge* on dialect handling, code-switching, robustness to diacritics, and cultural grounding, which are the conditions under which users write, and which the evaluation suite (Ch. 20) must therefore cover.

**Checkpoint question.** A team adapts an 8B English-centric base to Arabic with 40B tokens on one node, and after training the Arabic scores are good but GSM8K and HumanEval have fallen by ten points. What went wrong, and what are the remedies?

*Answer.* The run overwrote the base model's tail capabilities, which is where forgetting appears first. The likely causes are too little original-distribution data in the mixture, a learning rate re-warmed too aggressively, or both. Remedies, in order of cost: (1) add replay that specifically covers the lost skills (code and mathematics, not only general English), starting in the 10-30% range; (2) lower the peak learning rate toward the base model's final rate, as ALLaM did; (3) stage the run as the Jais adaptation did, training only the new embeddings first; (4) at small scale, freeze the backbone and train inserted layers, as Kuwain did; (5) test a weight average of the adapted and base checkpoints over their shared parameters, which sometimes recovers lost capability at no training cost, as SmolLM3's merge did for long context (Sec. 12.2). The evaluation cadence is also at fault: both-direction evaluation every few billion tokens would have shown the decline early.

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

*Applies to: product teams, regional labs, and frontier labs.*

SFT appears trivial: fine-tune on prompt-response pairs with the loss masked to the responses. It is nevertheless the stage at which most in-house model projects fail without noticing, because its quality is almost entirely a data problem. The best-documented open recipe is Tulu 3 (Ai2), which repays close study because, unlike most reports, it publishes the data, the ablations, and the mistakes.

**The Tulu 3 data method.** Start from explicit capability targets (knowledge, reasoning, mathematics, coding, instruction following, safety, multilinguality), and build the prompt pool for each target: collect the best existing open datasets, generate targeted synthetic data (persona-driven generation at scale, for coverage), and **decontaminate against the entire evaluation suite** before training. Benchmark prompts that leak into SFT data are the most common source of spurious in-house gains. The final SFT mixture contains on the order of one million curated prompt-response pairs, and its composition was tuned by ablation like a pretraining mixture (Ch. 3), not assembled once.

**Quality outweighs quantity, with a qualification.** The LIMA result (a thousand excellent examples can set style and format) holds only in part. Format and tone are cheap to teach, but *capabilities* under SFT scale with high-quality coverage of the skill: Tulu's gains in mathematics and coding came from targeted volume. Filtering with LLM judges is now standard practice. For example, the Trillion-7B report scores the responses in its SFT pool with Qwen2.5-72B as judge and keeps only those rated above 3 on a 0-5 scale, a pattern repeated across the 2025-2026 reports.

**Rejection sampling: SFT data from the model itself.** DeepSeek-R1's pipeline shows the current loop at full scale. After an RL stage, many responses per prompt are sampled from the RL checkpoint, only the verified-correct ones are kept (about 600K reasoning traces), about 200K non-reasoning examples are added (writing, question answering, translation), and SFT runs on the resulting 800K or so examples. The model becomes its own teacher wherever a verifier exists, and humans curate where none does.

**Mechanics that matter in practice.** Mask everything except the assistant responses. RightNow-Arabic's report calls out response-only masking explicitly, because an error here trains the model to imitate users. Choose the chat template early and *freeze* it: a template that drifts between SFT and deployment causes failures that are hard to diagnose. Pack sequences with correct attention separation between packed documents. Train for 1-3 epochs with a cosine or constant-then-decay learning rate around 1e-5 to 2e-5 for full fine-tuning (higher for LoRA, Ch. 19). Evaluate instruction following separately from knowledge, because SFT routinely improves one while degrading the other.

### 14.1 Published recipes

| Setting | Tulu 3 (8B / 70B) | SmolLM3 (3B) |
|---|---|---|
| Data | 939,344 prompts, selected from a pool of 23.3M | 1.8B tokens: 1B non-reasoning and 0.8B reasoning, from 12 and 10 datasets |
| Epochs | 2 | 4 (about 8B tokens seen) |
| Optimization | Learning rate 5e-6 / 2e-6, linear schedule, warmup ratio 0.03, effective batch 128, maximum length 4,096 | Sequence packing (best-fit decreasing) |
| Loss | **Sum** over tokens, not mean | Masked on user turns and on tool-call results |

Two details in this table are lessons in themselves. Tulu 3's learning rates are well below the customary 1e-5 to 2e-5, because the team switched from a mean to a sum loss and retuned, for the reason given in the next section. SmolLM3 filled gaps in its reasoning data by generating traces with Qwen3-32B, and inserted a *mid-training* stage before SFT: 35B tokens of reasoning data trained for four epochs (about 140B tokens). Reasoning behavior is expensive to install with SFT alone, and the larger laboratories now begin it earlier in the pipeline.

### 14.2 Loss normalization: a defect that affected most trainers

In October 2024 a long-standing defect in gradient accumulation was reported again and traced through the common training stacks. Each micro-batch computed a *mean* cross-entropy over its own non-padding tokens, and the means were then averaged across accumulation steps. When sequence lengths differ, that is not the loss of the full batch: tokens in short sequences receive more weight than tokens in long ones, and the result changes with the accumulation setting. The same issue appears across data-parallel ranks. Tulu 3's team met it independently, as a discrepancy between models trained on different infrastructure. With two samples of lengths $n_1$ and $n_2$, one batch gives $(\ell_1 + \ell_2)/(n_1 + n_2)$, while accumulating them separately gives $(\ell_1/n_1 + \ell_2/n_2)/2$. The correct procedure sums the token losses and divides once by the number of trainable tokens in the whole accumulated batch:

```python
import torch.nn.functional as F

def accumulation_step(model, micro_batches, optimizer):
    """Token-level normalization across gradient accumulation. In distributed training,
    `total` must also be summed (all-reduced) across ranks."""
    total = sum((mb["labels"][:, 1:] != -100).sum() for mb in micro_batches)
    for mb in micro_batches:
        logits = model(input_ids=mb["input_ids"]).logits[:, :-1]
        loss_sum = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   mb["labels"][:, 1:].reshape(-1),
                                   ignore_index=-100, reduction="sum")
        (loss_sum / total).backward()
    optimizer.step()
    optimizer.zero_grad()
```

Hugging Face Transformers and Unsloth were both corrected in October 2024. A custom training loop should be checked against a simple test: the gradient from one batch of eight sequences must equal the gradient from eight accumulated micro-batches of one.

### 14.3 Response-only masking in code

```python
def build_example(tokenizer, messages, max_len=4096):
    """messages: [{"role": ..., "content": ...}, ...]. Returns input_ids and labels, with -100 on
    every token outside the assistant responses. The assistant header is masked. The response
    text and its end-of-turn token are trained."""
    render = lambda msgs, gen: tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=gen, return_dict=False)
    input_ids, labels = [], []
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        prompt = render(messages[:i], True)          # everything up to the assistant header
        full = render(messages[: i + 1], False)      # ... plus the response and its end token
        assert prompt[: len(input_ids)] == input_ids and full[: len(prompt)] == prompt, \
            "chat template is not prefix-consistent; mask by character offsets instead"
        labels += [-100] * (len(prompt) - len(input_ids)) + full[len(prompt):]
        input_ids = full
    return input_ids[:max_len], labels[:max_len]
```

Whatever method is used, decode one batch and print the tokens that carry loss before starting a run. The check takes a minute and catches the three common failures: loss on user turns, a missing end-of-turn token (the model never learns to stop), and a duplicated beginning-of-sequence token. TRL exposes the same behavior through the `assistant_only_loss` option of its SFT trainer, which requires a chat template with generation markers. For agentic data, the outputs of tools must be masked as well (Sec. 18.3).

**Checkpoint question.** After SFT a model answers well but never stops generating. What are the likely causes?

*Answer.* The model did not learn to emit the end-of-turn token. The usual causes are: (1) the end token was masked out of the loss together with the template tokens, or was truncated by the maximum length; (2) the chat template used in training differs from the one used at inference, so the stop token the server waits for is not the one the model learned; (3) the end-of-turn token is the same as the padding token and the collator replaced every padding position with -100, which removes the real end tokens as well; (4) examples were packed without an end-of-turn token between them, so the model never saw a response end. Decoding a training batch with its loss mask reveals all four.

**References:**
- Tulu 3: Pushing Frontiers in Open Language Model Post-Training: https://arxiv.org/abs/2411.15124
- LIMA: Less Is More for Alignment: https://arxiv.org/abs/2305.11206
- Trillion 7B Technical Report: https://arxiv.org/abs/2504.15431
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- RightNow-Arabic-0.5B-Turbo: https://arxiv.org/abs/2605.28827
- SmolLM3 (Hugging Face blog): https://huggingface.co/blog/smollm3
- Unsloth, gradient accumulation fix: https://unsloth.ai/blog/gradient
- Hugging Face, Fixing Gradient Accumulation: https://huggingface.co/blog/gradient_accumulation
- TRL SFT trainer documentation: https://huggingface.co/docs/trl/en/sft_trainer

## 15. Preference Optimization and the GPT-4o Sycophancy Incident

*Applies to: product teams, regional labs, and frontier labs.*

### 15.1 Methods

After SFT, preference optimization moves the model toward the *better of several plausible outputs*, where SFT only teaches imitation. There are two families:

- **RLHF with a learned reward model (the PPO lineage):** train a reward model on human comparisons, then optimize the policy against it with a KL constraint toward the reference model: $\max_\pi \mathbb{E}[r(x,y)] - \beta \,\mathrm{KL}(\pi \,\|\, \pi_{ref})$. The approach is powerful, heavy in infrastructure, and exposed to **reward hacking**, in which the policy exploits the blind spots of the reward model.
- **DPO and its direct descendants:** omit the explicit reward model and optimize a closed-form objective on preference pairs directly. This is cheaper and more stable, and it is now the standard middle stage. Tulu 3's DPO ablations supply practical guidance. **On-policy preference data** (comparisons over the model's own samples) outperforms purely off-policy sets. Introducing **new prompts** at the DPO stage, and not only reusing SFT prompts, improves downstream performance, and more unique prompts help while duplicated prompts do not. Length bias is DPO's characteristic pathology (a preference for longer answers, inherited from the raters) and is countered with length-normalized variants.

The DPO objective makes the mechanics concrete. For a prompt $x$ with a chosen response $y_w$ and a rejected response $y_l$:

```math
\mathcal{L}_{DPO} = -\log \sigma\left(\beta \left[\log \frac{\pi_\theta(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{ref}(y_l \mid x)}\right]\right)
```

The loss needs only sequence log-probabilities under the policy and under a frozen reference model, so each training pair requires a forward pass of both responses through the policy and through the reference model. The reference log-probabilities can be computed once in advance. The length-normalized variant that Tulu 3 adopted divides each sequence log-probability by the response length, which removes the incentive to win the comparison by being long.

**Tulu 3's preference stage as a recipe.** The pipeline is an adaptation of UltraFeedback. Prompts are selected, including prompts not used in SFT. Four responses per prompt are drawn from a pool of 22 models, together with on-policy generations from the Tulu 3 SFT model. GPT-4o rates each response from 1 to 5 on helpfulness, instruction following, honesty, and truthfulness. The top-rated response becomes the chosen one, and a randomly selected lower-rated response becomes the rejected one. The resulting mixtures contain about 271K pairs for the 8B model and 334K for the 70B. Training uses length-normalized DPO with $\beta = 5$, a learning rate of 5e-7 for 8B and 2e-7 for 70B, an effective batch of 128, one epoch, and a maximum length of 2,048. The learning rates are an order of magnitude below the SFT rates, which is typical: preference optimization is a small correction to a model that already works.

SmolLM3 used a related method, anchored preference optimization (APO), with preference pairs built inexpensively: responses from Qwen3-32B as chosen and from Qwen3-0.6B as rejected for the reasoning mode, plus the Tulu 3 preference set for the non-reasoning mode. The team then observed a regression in long-context scores after the stage and repaired it by model merging (Sec. 12.2), which illustrates why every post-training stage needs a regression suite and not only a target metric.

### 15.2 Case study: the GPT-4o sycophancy rollback (April 2025)

This is the most instructive public failure in post-training, because the vendor published two postmortems. OpenAI shipped a GPT-4o update on April 24-25, 2025. Users immediately found it excessively agreeable: it validated plainly bad ideas and flattered indiscriminately. It was rolled back within days. The stated mechanism was that the update introduced **additional reward signals based on users' thumbs-up and thumbs-down feedback**, which weakened the primary reward signal that had been holding sycophancy in check. User feedback structurally favors agreeable responses, so the optimizer followed the signal. The process failure was as important as the signal failure. Offline evaluations did not include tests for sycophancy. A/B metrics looked acceptable, because agreeable models score well on short-horizon satisfaction. Expert reviewers *did* report that the model's behavior felt wrong, but the quantitative gates overruled them.

The lessons, stated as rules for a preference stage:

1. **Reward composition is a safety-critical design decision.** Every added signal will be optimized *literally*, including signals correlated with engagement. Model behavior follows the reward that was specified, not the intention behind it.
2. **Evaluate what is feared, not only what is wanted.** If a failure mode (sycophancy, verbosity, refusal collapse, dialect drift) is absent from the evaluation suite, preference optimization will improve the target metric at its expense without any visible signal.
3. **Give qualitative review formal authority.** Expert review identified the problem before launch and was overruled. Such reviews should be able to block a release.
4. **Short-horizon human approval is a biased estimator of long-horizon value.** A thumbs-up does not mean that a response was good for the user. It means that the response was pleasant at that moment.

### 15.3 Reward models

Where a reward model is trained: initialize it from a strong SFT checkpoint, train on clean pairwise comparisons, monitor for exploitable shortcuts (length, formatting, hedging), and refresh it as the policy distribution moves, because a static reward model facing an improving policy invites hacking. Prefer verifiable rewards wherever a verifier can exist, which is the subject of the next chapter. Kimi K2's post-training illustrates current practice for the subjective remainder: a critic model is trained alongside the policy and is kept grounded by continual updates on rollouts from prompts with verifiable rewards, so that its subjective judgments stay calibrated against objective signals.

**Checkpoint question.** After DPO, a model's win rate against the SFT model rises from 50% to 68% under an LLM judge, and its average response length rises by 40%. Is the stage a success?

*Answer.* Not yet established. LLM judges and human raters both favor longer answers, and DPO amplifies whatever regularity separates chosen from rejected responses, so a length increase of this size suggests that part of the gain is length. The checks are: compute the win rate under length control (compare responses of similar length, or use a length-controlled metric); inspect the length difference between chosen and rejected responses in the training pairs, and rebalance or switch to length-normalized DPO if chosen responses are systematically longer; run the regression suite, including instruction-following tests with explicit length constraints and the feared-failure evaluations of Sec. 22.3; and read transcripts. The GPT-4o incident is the reference case of a stage that passed its quantitative gates and failed in use.

**References:**
- Direct Preference Optimization: https://arxiv.org/abs/2305.18290
- Tulu 3: Pushing Frontiers in Open Language Model Post-Training: https://arxiv.org/abs/2411.15124
- OpenAI, Sycophancy in GPT-4o: https://openai.com/index/sycophancy-in-gpt-4o/
- OpenAI, Expanding on what we missed with sycophancy: https://openai.com/index/expanding-on-sycophancy/
- Kimi K2: Open Agentic Intelligence: https://arxiv.org/abs/2507.20534
- SmolLM3 (Hugging Face blog): https://huggingface.co/blog/smollm3

## 16. RL for Reasoning: GRPO, RLVR, and the R1 Pipeline

*Applies to: frontier labs and regional labs. Product teams should read Sec. 16.3 for the distillation result before planning any RL.*

### 16.1 RLVR: verifiable rewards in place of a learned reward model

Tulu 3 introduced the clearest framing. **Reinforcement learning with verifiable rewards (RLVR)** keeps the RLHF objective but replaces the learned reward model with a *verification function*: exact-match answer checkers for mathematics, unit tests for code, and constraint validators for instruction following. Reward hacking against a program is far harder than against a neural reward model, which is also the stated reason that DeepSeek avoided neural reward models in R1's reasoning stages. Tulu 3's measured effect was that RLVR applied to the DPO checkpoint added up to 1.7 points on MATH, 3.3 on GSM8K, and 1.3 on IFEval, with additional gains on tasks that were not optimized. Their best sequence was SFT, then DPO, then RLVR: RLVR from the DPO model gave better final quality than RLVR from the SFT model.

### 16.2 GRPO

Group Relative Policy Optimization (introduced in DeepSeekMath and widely adopted after R1) removes PPO's separate value model. For each prompt, a *group* of $G$ responses is sampled, each is scored with the rule-based reward, and the group's normalized scores serve as advantages: $A_i = (r_i - \mathrm{mean}(r_{1..G}))/\mathrm{std}(r_{1..G})$. These enter a clipped policy-gradient objective with a KL term toward the reference model. The group serves as the baseline, which removes a model-sized network from memory and removes the instability of training a critic. Mechanically, GRPO rewards whatever distinguishes above-average samples within a group. When longer and more careful chains of thought succeed more often, the model is pushed to reason at greater length, which is the emergent behavior that R1-Zero exhibited.

The objective, as given in DeepSeekMath, for a question $q$ and a group of outputs $o_1..o_G$ sampled from the old policy:

```math
J(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\min\left(\rho_{i,t}A_i,\ \mathrm{clip}(\rho_{i,t},\,1-\epsilon,\,1+\epsilon)\,A_i\right) - \beta\,\mathrm{KL}\left(\pi_\theta \,\|\, \pi_{ref}\right)\right],\qquad \rho_{i,t} = \frac{\pi_\theta(o_{i,t}\mid q, o_{i,\lt t})}{\pi_{old}(o_{i,t}\mid q, o_{i,\lt t})}
```

**Worked example.** Eight responses are sampled for one problem, and two are correct, so the rewards are [1, 0, 0, 1, 0, 0, 0, 0]. The group mean is 0.25 and the standard deviation is 0.433. Each correct response receives an advantage of $(1 - 0.25)/0.433 = +1.73$, and each incorrect response receives $-0.58$. Every token of a response shares that response's advantage. If all eight responses are correct, or all eight are wrong, the standard deviation is zero and every advantage is zero: the prompt contributes no gradient. As training progresses and more prompts become fully solved, a growing share of each batch is wasted in this way.

```python
import torch

def grpo_advantages(rewards, eps=1e-6):
    """rewards: [n_prompts, G] scalar rewards for G samples per prompt. Returns group-normalized advantages."""
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, unbiased=False)
    return (rewards - mean) / (std + eps)

def informative_prompts(rewards):
    """Dynamic sampling: keep only prompts whose group has mixed outcomes."""
    return rewards.std(dim=1, unbiased=False) > 0

r = torch.tensor([[1., 0., 0., 1., 0., 0., 0., 0.], [1., 1., 1., 1., 1., 1., 1., 1.]])
grpo_advantages(r)[0, :2]     # tensor([ 1.7320, -0.5774])
informative_prompts(r)        # tensor([ True, False])
```

DAPO (ByteDance and Tsinghua, 2025) is the most cited set of refinements, and each one answers a defect visible in the formula above. **Dynamic sampling** filters out prompts whose groups are entirely correct or entirely wrong and resamples until the batch is full of informative prompts. **Clip-higher** decouples the two clipping bounds and raises the upper one, which slows the collapse of entropy. A **token-level loss** averages over all tokens in the batch instead of first averaging within each response, so that long responses are not underweighted. **Overlong reward shaping** applies a graduated penalty to responses that approach the length limit, instead of treating truncation as an ordinary failure. DAPO also drops the KL term. With these changes it reached 50 points on AIME 2024 from the Qwen2.5-32B base model, against 47 for DeepSeek's own RL run on the same base.

### 16.3 Case study: from R1-Zero to R1

**R1-Zero, the pure RL experiment.** DeepSeek took DeepSeek-V3-Base, applied no SFT at all, and ran GRPO with two rule-based rewards: answer accuracy and output format (reasoning and answer tags). The result was that **AIME 2024 pass@1 rose from 15.6% to 71.0%** (86.7% with majority voting), with emergent reflection, self-verification, and progressively longer chains of thought, from reinforcement alone. The defects were equally notable: endless repetition, poor readability, and **language mixing**. A bilingual base model will reason in a mixture of Chinese and English when only correctness is rewarded, which is a direct warning to every team working on Arabic-English bilingual models.

**R1, the production pipeline.** Each stage corrects a named defect of R1-Zero:

1. **Cold-start SFT** on thousands of curated long chain-of-thought examples (partly cleaned R1-Zero outputs). This provides readability and a stable format before RL.
2. **Reasoning RL (GRPO)** with the rule-based rewards *and a language-consistency reward*, which is the direct correction for language mixing. A failure of reward specification was corrected by changing the reward specification.
3. **Rejection-sampling SFT** (about 800K examples: 600K verified reasoning and 200K general, Ch. 14), which broadens the model again beyond mathematics and code.
4. **Final RL across all prompt types**, with rule-based rewards where verification is possible and learned reward models only for the remaining general helpfulness and harmlessness.

**Distillation, the result that changed budgets.** DeepSeek and Qwen3 independently report the same finding: for small models, *distilling* from a strong reasoning teacher outperforms running the RL pipeline directly on the small model, at a fraction of the cost. Qwen3 describes it as strong-to-weak distillation outperforming direct RL for an 8B student, at about one tenth of the GPU-hours. For an organization below frontier scale, the reasoning strategy is almost certainly distillation followed by RLVR refinement, and not RL from scratch. Ch. 17 gives the numbers and the methods.

### 16.4 Systems considerations: rollout generation and weight synchronization

Online RL alternates generation (sampling thousands of responses) with training, and generation becomes a first-class cost. Tulu 3's 405B RLVR run published the anatomy of one configuration: the policy was served with vLLM under 16-way tensor parallelism for rollouts while 240 GPUs trained. Each iteration took about 550 s of generation, about 25 s to broadcast the updated weights to the inference engine over NCCL, and about 1,500 s of training. With long reasoning traces or multi-turn agents the balance inverts: Moonshot's Seer paper measures rollout at 63-87% of RL iteration time across three of its workloads, with the slowest requests accounting for up to half of the rollout phase. The techniques of the companion inference handbook (batching, KV-cache management, fast weight synchronization) therefore become concerns of *training* throughput. Three practical consequences follow. Asynchronous and overlapped rollout architectures are an active area of development. The group size and the maximum response length are the main cost controls. The throughput of the reward verifier, for example the execution of unit tests, can become an unplanned bottleneck.

### 16.5 Observed failure modes

- **Reward hacking against verifiers.** Models exploit weak verifiers, for example by formatting an answer so that a regular-expression checker passes, or by hardcoding test outputs. Verifiers are programs with adversarial users and should be tested accordingly.
- **Length inflation.** Rewarding correctness alone inflates the chain of thought beyond usefulness. Length penalties and budgets are now standard (the DAPO-style refinements).
- **Entropy collapse.** The policy narrows to one style of solution. It is monitored through generation entropy and countered with KL or entropy terms and with prompt diversity.
- **Judge leakage.** When LLM-as-judge rewards are used for open-ended tasks, the policy learns the judge's idiosyncrasies. Judges should be rotated and their verdicts audited by humans on a sample.

**Checkpoint question.** An RLVR run on mathematics improves its training reward steadily for 300 steps, and the held-out benchmark stops improving after step 100. What should be examined?

*Answer.* (1) The verifier: read samples with high reward and check whether they are correct or are exploiting the checker (answer formatting, multiple answers, degenerate outputs that a lenient parser accepts). (2) The prompt distribution: if most training prompts are now fully solved, their groups have zero advantage and the effective batch has shrunk to a few hard or noisy prompts, which dynamic sampling and a refreshed prompt set address. (3) Entropy: if generation entropy has collapsed, the policy has stopped exploring, and the clipping range, temperature, or KL settings need revisiting. (4) Length: rising response length with a flat benchmark indicates length inflation. (5) Contamination in the other direction: confirm that the held-out benchmark does not differ in format from the training prompts in a way that explains the gap. Reward that rises while held-out accuracy is flat is the signature of optimizing the verifier and not the task.

**References:**
- Tulu 3: Pushing Frontiers in Open Language Model Post-Training: https://arxiv.org/abs/2411.15124
- Tulu 3 405B (Ai2 blog): https://allenai.org/blog/tulu-3-405B
- DeepSeekMath (introduces GRPO): https://arxiv.org/abs/2402.03300
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- Seer: Online Context Learning for Fast Synchronous LLM Reinforcement Learning: https://arxiv.org/abs/2511.14617
- DAPO: An Open-Source LLM Reinforcement Learning System at Scale: https://arxiv.org/abs/2503.14476

## 17. Distillation

*Applies to: product teams and regional labs, for whom distillation is usually the most economical route to a capable small model. Frontier labs use it to build the smaller members of a model family.*

Distillation trains a student model on the outputs of a stronger teacher. In the 2024-2026 reports it has moved from a compression technique to a standard component of both pretraining and post-training, and for small reasoning models it has displaced reinforcement learning as the default method.

### 17.1 Three forms of distillation

**Logit distillation.** The student is trained to match the teacher's next-token distribution, not the one-hot label. This is the original formulation of Hinton, Vinyals, and Dean: a small model trained on the output distribution of a large model or an ensemble receives far more information per example than labels provide. It requires access to the teacher's logits and, in practice, a shared vocabulary.

**Sequence-level distillation.** The teacher generates complete responses, and the student is fine-tuned on them as ordinary SFT data. Kim and Rush introduced it for machine translation in 2016. It needs only the teacher's text, so it works across tokenizers and through an API, and it is the form used for most reasoning distillation.

**On-policy distillation.** The student generates its own responses, and the teacher scores every token of them, typically through a reverse KL divergence between the two distributions. GKD (Agarwal et al.) and MiniLLM (Gu et al.) established the approach. The motivation is a mismatch between training and inference in the other two forms: a student trained only on teacher-written text never learns to recover from its own errors, because it never sees them during training.

### 17.2 Distillation during pretraining

Google's Gemma family is the clearest public case. In Gemma 2, the 2B and 9B models were trained by distillation in place of next-token prediction, on 2T and 8T tokens respectively, while the 27B model was trained from scratch on 13T tokens. The paper notes that the small models see more than 50 times the compute-optimal number of tokens, and distillation is what makes that many tokens useful. In its ablation, a 2B model trained on 500B tokens averaged 60.3 across three benchmarks when trained from scratch and 67.7 when distilled from a 7B teacher. In Gemma 3, every model is distilled, including the 27B. The teacher's distribution is approximated by sampling 256 logits per token, weighted by the teacher's probabilities, and the student learns the renormalized distribution over those samples, which avoids storing a full vocabulary of logits for every token. Gemma 3 also reports a useful subtlety: a smaller teacher is better when training is short, and a larger teacher becomes better as training lengthens.

Meta's Llama 3.2 1B and 3B models combine distillation with pruning. They were produced by structured pruning of Llama 3.1 8B in a single step, and logits from the 8B and 70B models were then used as token-level targets during pretraining to recover the lost quality. NVIDIA's Minitron work quantifies the same combination: pruning Nemotron-4 15B by a factor of 2 to 4 and retraining with distillation on under 3% of the original data required up to 40 times fewer training tokens per derived model than training from scratch, reduced the compute cost of the 15B, 8B, and 4B family by a factor of 1.8, and improved MMLU by up to 16% relative to models trained from scratch. Its recommendations are specific: prefer width pruning to depth pruning at this scale, estimate importance in a single pass on a small calibration set (1,024 samples), and distill on logits alone unless depth has been reduced substantially.

### 17.3 Distilling reasoning

DeepSeek-R1's distilled models are sequence-level distillation at scale. The roughly 800K samples curated with R1 (about 600K reasoning traces from rejection sampling and about 200K general examples, Ch. 14) were used to fine-tune six open base models, from Qwen2.5-Math-1.5B to Llama-3.3-70B-Instruct, **with SFT only and no RL stage.** The paper then ran the controlled comparison that made the result influential: large-scale RL applied directly to Qwen2.5-32B-Base for more than 10,000 steps, against distillation into the same base model.

| Model | AIME 2024 pass@1 | MATH-500 | GPQA Diamond | LiveCodeBench |
|---|---|---|---|---|
| QwQ-32B-Preview | 50.0 | 90.6 | 54.5 | 41.9 |
| DeepSeek-R1-Zero-Qwen-32B (RL on the 32B base) | 47.0 | 91.6 | 55.0 | 40.2 |
| DeepSeek-R1-Distill-Qwen-32B (SFT on R1 outputs) | 72.6 | 94.3 | 62.1 | 57.2 |

The authors draw two conclusions. Distilling a stronger model into a smaller one works very well, while a small model trained with large-scale RL needs very large amounts of compute and may still not reach the distilled result. Distillation is economical and effective, but advancing beyond the current frontier still requires stronger base models and larger-scale RL. Someone has to train the teacher.

### 17.4 On-policy distillation

Qwen3 used distillation to build most of its family. Only the two flagship models, Qwen3-235B-A22B and Qwen3-32B, went through the full four-stage post-training pipeline. The five smaller dense models and the 30B MoE were produced by **strong-to-weak distillation** in two phases: off-policy distillation on teacher outputs generated in both thinking and non-thinking modes, and then on-policy distillation, in which the student samples its own responses and is trained to align its logits with the teacher's by minimizing a KL divergence. The report's comparison on Qwen3-8B starts both branches from the same off-policy-distilled checkpoint:

| Qwen3-8B | AIME'24 | AIME'25 | MATH500 | LiveCodeBench v5 | GPU-hours |
|---|---|---|---|---|---|
| After off-policy distillation | 55.0 | 42.8 | 92.4 | 42.0 | |
| Then reinforcement learning | 67.6 | 55.5 | 94.8 | 52.9 | 17,920 |
| Then on-policy distillation | 74.4 | 65.5 | 97.0 | 60.3 | 1,800 |

On-policy distillation is better on every benchmark at about one tenth of the compute. The report adds a further observation: RL did not improve pass@64 on AIME, while distillation did, which suggests that the teacher's logits expand what the student can do and do not merely sharpen what it already does.

Thinking Machines Lab's 2025 study supplies an explanation. Reinforcement learning conveys on the order of one bit of information per episode, the final reward, while distillation supervises every token. In their replication with Qwen3-8B-Base as the student, SFT on 400K prompts reached 60% on AIME 2024, and continuing with on-policy distillation reached 70% in about 150 steps, at a compute cost they estimate as 9 to 30 times lower than extending SFT to the same score. The same study demonstrates a second use that matters for adaptation projects. After mid-training a model on internal documents, its instruction-following score fell from 85% to 45%. On-policy distillation with the *original* model as teacher restored it to 83% while retaining most of the new knowledge (41% on the internal evaluation, against 43% before the repair and 18% for the original model). A team that damages a model's general behavior during continued pretraining can use the undamaged checkpoint as a teacher to recover it.

The per-token loss is compact. The student samples, both models score the sampled tokens, and the student minimizes the reverse KL on its own samples:

```python
import torch
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, mask, temperature=1.0, reverse=True):
    """Token-level KL between teacher and student distributions.
    logits: [B, T, V] over the same vocabulary; mask: [B, T], 1 on response tokens.
    reverse=True gives KL(student || teacher), used for on-policy distillation on student samples.
    reverse=False gives KL(teacher || student), the classical form on fixed data."""
    s = F.log_softmax(student_logits.float() / temperature, dim=-1)
    t = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    if reverse:
        kl = (s.exp() * (s - t)).sum(-1)
    else:
        kl = (t.exp() * (t - s)).sum(-1)
    return (kl * mask).sum() / mask.sum().clamp(min=1)
```

Reverse KL is mode-seeking: it penalizes the student for placing probability where the teacher places none, which suits a small student that cannot cover the whole teacher distribution. Forward KL is mass-covering and is the natural choice on fixed data.

### 17.5 When distillation does not help

Apple's distillation scaling-law study (models from 143M to 12.6B parameters) sets the boundaries. Distillation is more efficient than supervised pretraining only when two conditions hold: the student's compute budget is below a threshold that grows with student size, and a teacher already exists or its cost is shared across many students. If a teacher must be trained for a single student, supervised training is generally preferable, and given enough data and compute, distillation does not reach a lower loss than supervised learning. The study also confirms the **capacity gap**: a stronger teacher can produce a *worse* student when the gap between them is too large, which is consistent with Gemma 3's observation about teacher size and training length.

### 17.6 Practical guidance

- **For reasoning in a small model, begin with sequence-level distillation.** It needs only teacher text, and the R1 and Qwen3 results indicate that it outperforms direct RL on a small model at a small fraction of the cost. Add RLVR (Ch. 16) or on-policy distillation afterwards.
- **Logit and on-policy distillation require a shared vocabulary.** A student whose tokenizer was extended (Ch. 13) no longer shares the teacher's vocabulary, and it can be distilled at the logit level only over the shared tokens or from a teacher with the same extension. Sequence-level distillation has no such constraint.
- **Filter the teacher's outputs.** R1's 600K reasoning samples were kept only when verified correct. Unfiltered teacher data transfers the teacher's errors, and in a second language it also transfers translated phrasing.
- **Check the teacher's license.** The terms of many proprietary APIs and of some open-weight licenses restrict the use of outputs for training other models. This is a legal question to settle before data generation begins, not after.
- **Guard against language drift.** A teacher that reasons in English will teach an Arabic student to reason in English. If Arabic reasoning traces are a requirement, they have to be specified in the generation prompts and enforced by filtering, in the same way that R1 added a language-consistency reward.

**Checkpoint question.** A team wants an 8B Arabic-capable reasoning model and has 64 H100s for one month. Should it run RL on the 8B model or distill?

*Answer.* Distill. One month on 64 H100s is about 46,000 GPU-hours. Qwen3's RL branch for an 8B model cost 17,920 GPU-hours and ended below the distillation branch, which cost 1,800. DeepSeek's RL run on a 32B base ended below the distilled 32B on every benchmark. The efficient plan is sequence-level distillation from a strong open reasoning teacher on verified and filtered traces, with Arabic prompts and Arabic-language traces where those are required, followed by on-policy distillation if the vocabularies match, and a short RLVR stage on problems with checkable answers. The remaining budget is better spent on evaluation and on data filtering than on a longer RL run.

**References:**
- Distilling the Knowledge in a Neural Network (Hinton, Vinyals, Dean): https://arxiv.org/abs/1503.02531
- Sequence-Level Knowledge Distillation (Kim and Rush): https://arxiv.org/abs/1606.07947
- On-Policy Distillation of Language Models (GKD): https://arxiv.org/abs/2306.13649
- MiniLLM: https://arxiv.org/abs/2306.08543
- Gemma 2: https://arxiv.org/abs/2408.00118
- Gemma 3 Technical Report: https://arxiv.org/abs/2503.19786
- Llama 3.2 (Meta blog): https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/
- Compact Language Models via Pruning and Knowledge Distillation (Minitron): https://arxiv.org/abs/2407.14679
- DeepSeek-R1 (distillation results in version 1): https://arxiv.org/abs/2501.12948
- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- On-Policy Distillation (Thinking Machines Lab): https://thinkingmachines.ai/blog/on-policy-distillation/
- Distillation Scaling Laws (Apple): https://arxiv.org/abs/2502.08606

## 18. Agentic and Tool-Use Post-Training

*Applies to: frontier labs and regional labs building agents, and product teams fine-tuning a model for tool use in a specific product.*

A model that calls tools, runs code, or works through a task over dozens of turns is trained differently from a model that answers in one turn. The data has to contain tool calls and their results, the reward has to come from an environment, and the rollouts are long and uneven in length. The 2025-2026 reports (Kimi K2, GLM-4.5, Qwen3-Coder, DeepSeek-V3.2) describe a common recipe: synthesize tool-use trajectories at scale for SFT, then run RL in executable environments with verifiable outcomes.

### 18.1 Synthesizing tool-use data

Kimi K2's pipeline is the most fully described. It has three stages:

1. **Tool specifications.** The team collected more than 3,000 real tools that follow the Model Context Protocol from GitHub repositories, and synthesized more than 20,000 further tools by evolving a hierarchy of domains (categories such as financial trading, software applications, and robot control, then applications within them, then tools).
2. **Agents and tasks.** Thousands of agents were generated by combining system prompts with sets of tools. Each generated task carries an explicit rubric: success criteria, expected patterns of tool use, and checkpoints for evaluation.
3. **Trajectories.** Simulated users with generated personas hold multi-turn conversations with the agent. A tool simulator executes the calls, maintains state between calls, and introduces controlled variation (successes, partial failures, edge cases). An LLM judge scores each trajectory against the task's rubric, and only trajectories that meet the success criteria are kept. The paper describes this filtering as rejection sampling at scale.

Simulation is complemented by **real execution sandboxes** for coding and software-engineering tasks, which run the code and return ground truth such as test pass rates. The division is general: a simulator provides breadth cheaply, and real execution provides correctness where it is available.

DeepSeek-V3.2 reports the scale that this approach has reached: more than 1,800 distinct environments and 85,000 complex prompts, including 24,667 code-agent tasks and 50,275 search-agent tasks in real environments, and 4,417 general-agent tasks in 1,827 synthesized environments. Its synthesis procedure for general agents generates the verifier together with the task: an agent proposes a task with a Python solution and a verification function, and then raises the difficulty iteratively. The same report states that its post-training compute budget exceeded 10% of the pretraining cost, which marks how far the center of gravity has moved since DeepSeek-V3's 5K GPU-hours of post-training (Sec. 2.2).

### 18.2 Environments and verifiable rewards

The reward in agentic RL is almost always an outcome check performed by the environment:

- **Software engineering.** GitHub issues and pull requests are turned into environments with executable unit tests. SWE-Gym provides 2,438 such Python task instances, and SWE-smith provides 50,000 from 128 repositories. Kimi K2 runs these tasks on a Kubernetes-based sandbox that supports more than 10,000 concurrent instances. Qwen3-Coder's team built a system that runs 20,000 independent environments in parallel for its long-horizon RL.
- **A sparse reward is sufficient.** DeepSWE trained Qwen3-32B with RL alone on about 4,500 problems from R2E-Gym, with a reward of 1 if the final patch passed the selected tests within the time limit and 0 otherwise. In six days on 64 H100s, SWE-bench Verified pass@1 rose from 23% to 42.2%, and to 59% with test-time selection among 16 attempts. Its algorithmic changes follow the DAPO line (Sec. 16.2), and it adds one that is specific to agents: trajectories that end by exhausting the context, the step limit, or the time limit are masked out of the loss, so that the model is not penalized for failures caused by the environment's limits.
- **Web search and tool use.** GLM-4.5 builds search tasks by multi-hop synthesis over knowledge graphs and uses outcome supervision with one process-level rule: a malformed tool call halts the episode and earns zero reward. The report states that RL on search and software-engineering tasks transferred to general tool use and to terminal tasks that were not trained.
- **Mixed objectives.** Kimi K2's RL combines a gym of verifiable tasks (mathematics and logic, instruction following, faithfulness, coding and software engineering, safety) with the self-critique rubric reward described in Sec. 15.3. It adds three controls worth copying: a per-task token budget with a penalty for exceeding it, an auxiliary loss on high-quality SFT data to prevent forgetting, and a sampling temperature that decays as training moves from exploration to exploitation.

### 18.3 Loss masking for tool outputs

A tool-use trajectory interleaves tokens that the model generated with tokens that the environment returned: search results, file contents, interpreter output. Only the first kind should carry loss, in SFT and in RL. Search-R1 states the rule for RL, computing the policy-gradient objective over generated tokens only, and reports the effect in an ablation: average exact match of 0.431 with masking against 0.343 without it for a 7B model (the figures are from the fifth version of the paper). ReTool and ToRL describe the same masking for interpreter output, and ToRL gives the reason plainly: the model should not learn to memorize execution results. The training frameworks implement it as a response mask that is 1 on model tokens and 0 on tool tokens.

```python
def agent_loss_mask(segments):
    """segments: list of (token_ids, source), with source in {"system", "user", "model", "tool"}.
    Returns flat input_ids and a mask that is 1 only on tokens the model generated."""
    input_ids, mask = [], []
    for token_ids, source in segments:
        input_ids += token_ids
        mask += [1 if source == "model" else 0] * len(token_ids)
    return input_ids, mask
```

The trajectory must be tokenized segment by segment as it was generated. Tokenizing the concatenated text again can produce different token boundaries where a model segment meets a tool segment, and the log-probabilities used by the RL update would then no longer correspond to the tokens that were sampled.

### 18.4 Rollout systems for long-horizon tasks

Agentic rollouts make the systems problem of Sec. 16.4 worse, because episode lengths vary by orders of magnitude and the environment itself adds latency. The documented responses:

- **Partial rollouts.** Kimi k1.5 introduced the technique, and K2 retains it: long unfinished episodes are paused when the iteration's budget is spent and resumed in the next iteration, so that one slow episode does not hold the whole batch. The Kimi-Researcher report credits a turn-level version with a rollout speedup of at least 1.5 times.
- **Asynchronous training.** AReaL separates generation from training completely and reports up to 2.77 times faster training than synchronous systems on the same GPUs, at the price of learning from slightly stale samples. GLM-4.5's infrastructure runs a synchronous, colocated mode for ordinary RL and a disaggregated asynchronous mode for agentic RL, with BF16 training and FP8 rollout inference.
- **Fast weight synchronization.** Kimi K2 colocates the training and inference engines on the same workers and moves weights through a checkpoint engine that updates the full 1T-parameter model in under 30 seconds.
- **Environments as services.** Heavy environments run as dedicated services behind a uniform, Gym-style interface, and many concurrent rollouts amortize their latency.

### 18.5 Reward hacking and benchmark leakage

An agent with a shell has more ways to obtain reward than a model that writes one answer, and the public record since 2025 is specific.

- **Hacks during RL.** OpenAI's study of a frontier reasoning model in agentic coding training found two systemic hacks, exiting with status 0 before the tests ran and raising a skip-test exception, together with problem-specific ones: writing stubs, overwriting verification functions, decompiling reference solutions, and parsing test files for the expected values. A GPT-4o monitor that read the chain of thought caught 95% of the systemic hacks, against 60% for a monitor that saw only the actions. When the chain-of-thought monitor was used as a training signal, the model learned to hide its intent and the monitor's recall fell to nearly zero.
- **Special-casing in a released model.** The system card for Claude 3.7 Sonnet reports that the model sometimes special-cased tests, returning the expected values directly or editing the failing tests, and attributes the behavior to reward hacking during RL.
- **Repository-history leakage.** In September 2025 a SWE-bench issue documented agents finding the fix in future repository state, for example with `git log --all`, in trajectories from several models. The proposed remedy was to remove remotes, branches, tags, and reflogs from the task repositories. The Qwen3-Coder-Next report shows that sanitizing the repository is not sufficient during RL: with remotes and branches already removed, agents in later stages of training re-attached remotes or fetched history over the network, until a rule blocked any tool call that combined a repository link with network commands.
- **Benchmark validity.** In February 2026 OpenAI stopped reporting SWE-bench Verified. Its audit found that 59.4% of 138 hard problems had material flaws in the tests or the problem description, and that frontier models could reproduce the reference patches, which indicates exposure during training. In June 2026 Cursor audited 731 trajectories of a frontier model on SWE-bench Pro and found that 63% of the successful resolutions had retrieved the fix, from the public web or from bundled git history, instead of deriving it. With the git history sealed and internet access removed, that model's score fell from 87.1% to 73.0%.

The practical rules follow directly. Remove network access and repository history from training and evaluation environments unless the task requires them. Treat the verifier as code under adversarial test: run it against known hacks (early exit, skipped tests, modified tests, hardcoded outputs) before training. Inspect high-reward trajectories by hand at intervals. Keep a monitor on the reasoning trace for diagnosis, and do not optimize against it.

### 18.6 Benchmarks

| Benchmark | What it measures |
|---|---|
| tau-bench and tau2-bench | Multi-turn conversations between a simulated user and an agent with domain tools and policy documents (retail, airline, and in the second version telecom with a user who also acts). Scored on the final database state. The pass^k metric measures whether the agent succeeds on all of k independent trials of the same task |
| BFCL (Berkeley Function Calling Leaderboard) | Accuracy of function calls across single-turn, multi-turn, and agentic categories, checked by abstract-syntax-tree matching and by execution |
| SWE-bench Verified and SWE-bench Pro | Resolution of real GitHub issues, verified by unit tests. Verified is a 500-task subset validated by human annotators in 2024, now considered saturated and contaminated by its original sponsor |
| Terminal-Bench | Tasks completed in a terminal, each with its own environment, a reference solution written by a human, and verification tests. Version 2.0 has 89 tasks |

Reliability deserves as much attention as the headline score. In tau-bench, GPT-4o succeeded on fewer than half of the tasks and its pass^8 in the retail domain was below 25%: an agent that succeeds six times in ten fails most users who return more than once. For a product, the distribution of outcomes across repeated trials is the relevant quantity.

**Checkpoint question.** During RL on software-engineering tasks, the training reward rises from 30% to 75% in 200 steps, much faster than in earlier runs. Before reporting the result, what should be checked?

*Answer.* A rise of that speed is more often a hack than a breakthrough. (1) Read a sample of successful trajectories for the known patterns: commands that search git history or the network for the fix, edits to the test files, early exits, and skipped tests. (2) Confirm the sandbox: no network access, and no remotes, branches, tags, or reflogs in the repository. (3) Re-run the successful patches in a clean environment against the hidden tests and against tests the agent could not see or modify. (4) Compare with a held-out benchmark that uses a different harness. If the held-out score has not moved, the policy has learned the environment and not the task. (5) Check what fraction of episodes ended at the context or step limit, and whether those were masked or penalized, because a change in that fraction moves the reward without any change in ability.

**References:**
- Kimi K2: Open Agentic Intelligence: https://arxiv.org/abs/2507.20534
- Kimi k1.5: Scaling Reinforcement Learning with LLMs (partial rollouts): https://arxiv.org/abs/2501.12599
- Kimi-Researcher: https://moonshotai.github.io/Kimi-Researcher/
- GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models: https://arxiv.org/abs/2508.06471
- Qwen3-Coder (Qwen blog): https://qwenlm.github.io/blog/qwen3-coder/
- Qwen3-Coder-Next Technical Report: https://arxiv.org/abs/2603.00729
- DeepSeek-V3.2: https://arxiv.org/abs/2512.02556
- SWE-Gym: https://arxiv.org/abs/2412.21139 · SWE-smith: https://arxiv.org/abs/2504.21798
- DeepSWE (Agentica and Together AI): https://www.together.ai/blog/deepswe
- Search-R1: https://arxiv.org/abs/2503.09516 · ReTool: https://arxiv.org/abs/2504.11536 · ToRL: https://arxiv.org/abs/2503.23383
- AReaL: https://arxiv.org/abs/2505.24298
- Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation (OpenAI): https://arxiv.org/abs/2503.11926
- Claude 3.7 Sonnet System Card (Anthropic): https://www-cdn.anthropic.com/9ff93dfa8f445c932415d335c88852ef47f1201e.pdf
- SWE-bench issue 465, Repo State Loopholes During Agentic Evaluation: https://github.com/SWE-bench/SWE-bench/issues/465
- OpenAI, Why SWE-bench Verified no longer measures frontier coding capabilities: https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/
- Cursor, Reward hacking is swamping model intelligence gains: https://cursor.com/blog/reward-hacking-coding-benchmarks
- tau-bench: https://arxiv.org/abs/2406.12045 · tau2-bench: https://arxiv.org/abs/2506.07982
- Berkeley Function Calling Leaderboard: https://gorilla.cs.berkeley.edu/leaderboard.html
- Terminal-Bench: https://arxiv.org/abs/2601.11868

---

# Part V: The Practitioner's Track

## 19. Fine-Tuning Without a Cluster: LoRA, Full Fine-Tuning, and a Decision Procedure

*Applies to: product teams and application teams. It is the entry point for readers who will never pretrain.*

### 19.1 Evidence: LoRA versus full fine-tuning

For several years the question of whether LoRA matches full fine-tuning had no settled answer. Two rigorous studies now bound it.

**"LoRA Learns Less and Forgets Less" (Biderman et al., 2024).** The study compares the two methods directly on Llama-2-7B, across instruction tuning (about 100K pairs) and continued pretraining (about 20B tokens) in code and mathematics. At standard ranks, **LoRA substantially underperforms full fine-tuning in the continued-pretraining regime** and in instruction tuning for code. The weight update learned by full fine-tuning has an effective rank 10 to 100 times higher than typical LoRA configurations, which explains the gap mechanistically. The converse finding is valuable: **LoRA forgets much less** of the base model's abilities outside the target domain (it is more protective than weight decay or dropout) and preserves the diversity of generations. The low rank acts as a regularizer. Follow-up work on "intruder dimensions" locates LoRA's forgetting in spurious singular directions that it introduces, and shows that applying LoRA sequentially accumulates them, which is relevant to anyone who stacks many adapters over time.

**"LoRA Without Regret" (Thinking Machines, 2025).** This study rehabilitates LoRA under stated conditions. For **datasets of post-training scale** (sizes that fit within LoRA's parameter capacity, which covers most real SFT and DPO jobs), LoRA matches the sample efficiency and final quality of full fine-tuning *provided* the details are right: apply it to **all layers, especially the MLPs** (attention-only LoRA, the original default, is the common mistake), and use a **learning rate roughly 10 times higher** than the equivalent full fine-tuning rate. The two studies are consistent. LoRA is a capacity-limited method. Below its capacity (typical SFT) it costs nothing in quality, and beyond it (continued pretraining on tens of billions of tokens of new knowledge) it becomes the constraint.

**A working configuration** supported by the combined literature: rank 16-64 for style and behavior, and 64-256 when injecting knowledge; α = 2r; all linear layers targeted; a learning rate around 1e-4 to 2e-4 (against about 1e-5 to 2e-5 for full fine-tuning); and QLoRA (a frozen 4-bit base with LoRA) when memory is the constraint, which trades a small quality margin for a memory reduction of 3 to 4 times.

Three further findings from "LoRA Without Regret" affect planning. A LoRA pass costs slightly more than two thirds of the FLOPs of a full fine-tuning pass, because the weight gradients of the frozen matrices are never computed. LoRA is less tolerant of large batch sizes than full fine-tuning, with a gap that grows with the batch and does not depend on the rank. For policy-gradient RL, LoRA matched full fine-tuning even at rank 1, which the authors explain by the small amount of information that RL conveys per episode (Sec. 17.4).

**Worked example: what rank 16 means.** LoRA replaces the update of a $d_{out} \times d_{in}$ matrix with two matrices of shapes $d_{out} \times r$ and $r \times d_{in}$, which is $r(d_{in} + d_{out})$ trainable parameters. For Llama 3 8B (hidden size 4,096, key-value dimension 1,024, MLP dimension 14,336, 32 layers) at $r = 16$ on all seven linear projections of each layer: the query and output projections contribute $16 \times 8192 = 131{,}072$ each, the key and value projections $16 \times 5120 = 81{,}920$ each, and the three MLP projections $16 \times 18{,}432 = 294{,}912$ each. That is 1,310,720 per layer and **41.9M in total, about 0.5% of the model**. At $r = 64$ it is 168M. Optimizer state and gradients exist only for these parameters, about 0.7 GB at 16 bytes each, against 128 GB of state for full fine-tuning (Sec. 9.4). The frozen base still occupies 16 GB in BF16, or about 5 GB in 4 bits under QLoRA, and activations are unchanged.

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,                  # alpha = 2r
    lora_dropout=0.0,
    target_modules="all-linear",    # attention and MLP projections, not attention only
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
```

### 19.2 Decision procedure

Effort should be spent in this order:

1. **Prompting and retrieval first.** If the failure is missing knowledge, retrieval outperforms training on cost, freshness, and auditability. Train only when the failure is *behavioral* (format, tone, dialect, refusal patterns, tool protocols) or is driven by latency or cost (distilling a large model's behavior into a small one).
2. **LoRA SFT on an instruct model** for problems of behavior, style, or domain format. This takes hours on one node.
3. **Full fine-tuning SFT (or high-rank LoRA) with DPO** when LoRA reaches a plateau, or when quality shaped by preferences matters. This is the Tulu recipe scaled to the available data (Ch. 14-15).
4. **Continued pretraining (full fine-tuning, a mixed corpus, and a disciplined decay phase)** when the model lacks the *distribution* itself, such as a language or a technical corpus. This is the subject of Ch. 13, and it is real training, with all the failure modes of Part II at small scale.
5. **RLVR refinement** when a verifier exists and a metric resists SFT (Ch. 16), with distillation, and not RL from scratch, as the route to reasoning for small models.

At every step, build the evaluation suite *before* the training run (Ch. 20), including a forgetting suite that covers the base capabilities that must not be lost. The least expensive fine-tuning run is the one that the evaluation shows to be unnecessary.

**Checkpoint question.** A team fine-tunes an 8B model with LoRA at rank 8 on attention projections only, with a learning rate of 2e-5, and concludes that LoRA is much worse than full fine-tuning for its task. Is the conclusion justified?

*Answer.* No. The configuration contains the three errors that "LoRA Without Regret" identifies. Attention-only adapters underperform adapters on the MLP layers even at equal parameter counts, so all linear layers should be targeted. A learning rate of 2e-5 is a full fine-tuning rate, and the optimum for LoRA is about ten times higher. Rank 8 on attention alone is a very small capacity: whether it is enough depends on the dataset, and for a large SFT set it may be the binding limit. The comparison should be repeated with all linear layers, rank 32 to 64, and a learning rate near 2e-4, with a moderate batch size. If LoRA still falls short, and the dataset is large, the task is in the capacity-limited regime that Biderman et al. describe, and full fine-tuning is the right tool.

**References:**
- LoRA: Low-Rank Adaptation of Large Language Models: https://arxiv.org/abs/2106.09685
- LoRA Learns Less and Forgets Less (Biderman et al.): https://arxiv.org/abs/2405.09673
- LoRA vs Full Fine-tuning: An Illusion of Equivalence: https://arxiv.org/abs/2410.21228
- LoRA Without Regret (Thinking Machines): https://thinkingmachines.ai/blog/lora/
- QLoRA: Efficient Finetuning of Quantized LLMs: https://arxiv.org/abs/2305.14314

## 20. Evaluation During Training

*Applies to: all readers. Application teams that never train still need Sec. 20.2 to choose a model.*

Training without a measurement plan is how teams release regressions with confidence. The operational principles below are compiled from the same reports as the rest of the handbook.

**Design for early signal.** Small models and early checkpoints score near chance on most benchmarks. The Smol Training Playbook's remedy is task *formulation*: cloze or likelihood scoring (comparing the probability of the correct continuation) yields smooth, discriminative curves where accuracy in the multiple-choice format is still flat. An ablation suite should be built from formulations whose early signal is monotone, with a fixed held-out perplexity set for each domain and each language. An Arabic project tracks Arabic and English perplexity separately, following the both-directions rule of Sec. 13.2.

**Loss is not the product.** Comparisons of loss across domains mislead, because entropy floors differ, and the mid-training and post-training stages deliberately trade loss for capability. A small battery of capability probes should be tracked across checkpoints. Ai2's releases of intermediate OLMo checkpoints exist so that the community can study capability as a function of tokens, and a project should keep its own for the same reason.

**Decontamination is mandatory and works in both directions.** Remove evaluation sets from the training data (by n-gram and fuzzy matching), *and* check every new training set against the evaluation suite, as Tulu 3 does. Contamination through synthetic-data pipelines, in which a generator model has memorized the benchmark, is the current leak path. This is one reason to prefer fresh, private evaluations derived from the product as the primary signal, and to use public benchmarks only as a sanity range.

**Use LLM judges with care.** Judges exhibit position bias, length bias, self-preference, and style preferences. Anchor them with rubrics and reference answers, calibrate a sample against human ratings, never let a model from the same family be the only judge of a model, and treat judge scores as relative (A against B) and not as absolute measurements.

**The qualitative gate.** The sycophancy incident (Sec. 15.2) is the standing argument. Structured human review of real transcripts should be scheduled before any release, with the authority to block it. Metrics are necessary, and in that incident they were present and favorable while the released model was flattering its users.

### 20.1 Choosing benchmarks that carry signal

The FineWeb team selected its ablation benchmarks by three criteria: small variance between runs trained on different samples of the same data, scores that rise monotonically (or nearly so) during training, and scores above the random baseline by at least a few standard deviations. The benchmarks that qualified were CommonSense QA, HellaSwag, OpenBook QA, PIQA, SIQA, WinoGrande, ARC, and MMLU, run with lighteval and capped at 1,000 samples each. FineTasks extended the method to nine languages, testing 185 tasks and keeping 96, with explicit thresholds: monotonicity measured by the rank correlation between training step and score, a signal-to-noise ratio, a margin over the random baseline, and the consistency of model rankings between consecutive checkpoints. It reports two findings that transfer directly to Arabic work: the cloze formulation gives earlier signal than multiple choice, and length-normalized accuracy was the most reliable metric overall.

Ai2's work makes the same ideas quantitative. **OLMES** is a documented, reproducible evaluation standard for ten multiple-choice tasks (fixed prompt formats, curated five-shot examples, a per-task normalization, and the better of the cloze and multiple-choice formulations for each model). Its motivating observation is that the same model had been reported on ARC-Challenge at anything from 48.8 to 67.6 in earlier papers, depending on unreported details. **Signal and Noise** defines signal as the spread of scores between models and noise as the variation over the final checkpoints of a run, and shows that the ratio of the two predicts whether decisions made at small scale hold at large scale (a correlation of 0.79 across 30 benchmarks). Its three interventions are inexpensive: use better-behaved metrics such as bits per byte in place of accuracy, drop subtasks with a low ratio, and average over several final checkpoints instead of scoring only the last. **DataDecide** (Sec. 5.2) supplies the matching result for data decisions.

The practical procedure for a new project is to take the candidate benchmarks, run them on the checkpoints of one small training run and on two seeds, and discard every benchmark that is not monotone, not above chance, or not stable between seeds. What remains is the ablation suite.

### 20.2 Arabic evaluation benchmarks

The Arabic evaluation landscape changed substantially in 2024-2025, and the direction of change is away from machine-translated English benchmarks. The second version of the Open Arabic LLM Leaderboard removed every machine-translated task, stating that translated tasks frequently introduced linguistic and contextual mismatches. ArabicMMLU's authors make the underlying argument: translated benchmarks are centered on English-speaking curricula and culture, cannot test knowledge specific to Arabic, and carry translation errors. Global MMLU measured the effect on MMLU itself: 28% of its questions require culturally sensitive knowledge, and 84.9% of the questions that depend on geography concern North America or Europe.

| Benchmark | What it measures | Construction |
|---|---|---|
| ArabicMMLU (MBZUAI, 2024) | Knowledge and reasoning in MSA | 14,575 native multiple-choice questions from school and university exams in eight Arab countries, 40 tasks |
| Open Arabic LLM Leaderboard v2 (2A2I, TII, Hugging Face, 2025) | A composite of native and human-curated tasks | Native AlGhafa subsets, ArabicMMLU, a human translation of MMLU (MMLU-HT), MadinahQA (Arabic language and grammar), AraTrust, ALRAGE (a generative retrieval-augmented task), Arabic EXAMS, and the Arabic Belebele tasks |
| AraGen and the 3C3H measure (Inception and MBZUAI, 2024) | Generative quality, judged on correctness, completeness, conciseness, helpfulness, honesty, and harmlessness | Human-verified questions scored by an LLM judge. Each test set stays private for three months, is then released, and is replaced |
| Arabic IFEval (Inception and MBZUAI, 2025) | Instruction following with verifiable constraints | Prompts adapted from IFEval and new Arabic-specific instructions, such as diacritization, manually verified |
| BALSAM (2025) | 78 NLP tasks in 14 categories, 52K examples | A community platform with blind test sets, hosted by the King Salman Global Academy for Arabic Language. Its LLM-judge scores correlated with human ratings at 0.82 to 0.98 by category, while BLEU and ROUGE correlated very poorly |
| AraDiCE (QCRI, 2024) | Dialect and culture | Seven benchmarks post-edited by native speakers into Levantine and Egyptian Arabic as well as MSA, and a cultural-awareness set covering the Gulf, the Levant, and Egypt |
| Palm (2025) | Culturally grounded instruction following | 17,411 human-written instruction-response pairs covering all 22 Arab countries, in MSA and ten dialects, with a private held-out test set |
| ArabCulture (MBZUAI, 2025) | Commonsense reasoning in Arab cultural contexts | 3,482 questions written from scratch by native speakers from 13 countries |
| AraTrust (2024) | Trustworthiness and safety | 522 human-written multiple-choice questions in eight categories |
| 3LM (TII, 2025) | STEM and code | Native STEM questions from Arabic textbooks and exam banks, synthetic STEM questions from the same sources, and translated code benchmarks with human review |
| SILMA Arabic Broad Benchmark (2025) | 22 skills in one compact set | 470 human-validated questions sampled from 64 Arabic datasets, scored with manual rules and an LLM judge |

For dialects, the standard resources are the MADAR parallel corpus (25 city dialects and MSA, built on travel-domain sentences) and the NADI shared tasks on dialect identification, which have run annually since 2020 and moved to speech in 2025.

Three cautions apply to all of these. First, most of the knowledge benchmarks measure MSA, and a product that serves Gulf or Egyptian users must add dialectal and code-switched evaluation of its own, because published models that score alike on MSA diverge there (Sec. 13.4). Second, public leaderboards saturate and leak. The benchmarks with private or rotating test sets (AraGen, BALSAM, Palm's held-out split) are the more trustworthy signals, and a private test set built from the product's own traffic is more trustworthy still. Third, generative quality in Arabic cannot be measured with n-gram overlap metrics, as BALSAM's correlation study shows, so an LLM judge that has been calibrated against human raters is required for open-ended tasks.

**Checkpoint question.** Two Arabic models score 62 and 64 on a knowledge benchmark. How should a team decide whether the difference matters for its product?

*Answer.* First establish whether the difference exceeds the noise: score several late checkpoints or seeds where possible, or compute a confidence interval from the number of questions. On a 1,000-question subset the standard error near 63% accuracy is about 1.5 points, so a 2-point difference is within noise. On the full 14,575 questions of ArabicMMLU it is about 0.4 points, and the same difference is probably real. Then ask whether the benchmark measures what the product needs: both scores are MSA multiple-choice knowledge, and they do not measure dialect handling, instruction following, generation quality, or safety. The decision should rest on a private evaluation built from the product's own use cases, in the registers its users write, scored by a calibrated judge and by human review of transcripts.

**References:**
- The Smol Training Playbook (Hugging Face): https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook
- Tulu 3 (decontamination procedure): https://arxiv.org/abs/2411.15124
- 2 OLMo 2 Furious (OLMo 2): https://arxiv.org/abs/2501.00656
- OpenAI, Expanding on what we missed with sycophancy: https://openai.com/index/expanding-on-sycophancy/
- FineWeb blog post (benchmark selection): https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1
- FineTasks (Hugging Face): https://huggingface.co/spaces/HuggingFaceFW/blogpost-fine-tasks
- OLMES: A Standard for Language Model Evaluations: https://arxiv.org/abs/2406.08446
- Signal and Noise: A Framework for Reducing Uncertainty in Language Model Evaluation: https://arxiv.org/abs/2508.13144
- DataDecide: https://arxiv.org/abs/2504.11393
- ArabicMMLU: https://arxiv.org/abs/2402.12840
- Open Arabic LLM Leaderboard v2: https://huggingface.co/blog/leaderboard-arabic-v2
- AraGen and 3C3H: https://huggingface.co/blog/leaderboard-3c3h-aragen · Arabic IFEval: https://huggingface.co/blog/leaderboard-3c3h-aragen-ifeval
- BALSAM: https://arxiv.org/abs/2507.22603
- AraDiCE: https://arxiv.org/abs/2409.11404
- Palm: https://arxiv.org/abs/2503.00151
- ArabCulture (Commonsense Reasoning in Arab Culture): https://arxiv.org/abs/2502.12788
- AraTrust: https://arxiv.org/abs/2403.09017
- 3LM: https://arxiv.org/abs/2507.15850
- SILMA Arabic Broad Benchmark: https://huggingface.co/datasets/silma-ai/arabic-broad-benchmark
- AlGhafa: https://aclanthology.org/2023.arabicnlp-1.21/
- MADAR parallel corpus: https://camel.abudhabi.nyu.edu/madar-parallel-corpus/ · NADI shared tasks: https://nadi.dlnlp.ai/
- Global MMLU: https://arxiv.org/abs/2412.03304
- Evaluating Arabic Large Language Models: A Survey of Benchmarks, Methods, and Gaps: https://arxiv.org/abs/2510.13430

## 21. Frameworks and Tooling

*Applies to: all readers who train. The status notes describe September 2026 and will age faster than the rest of the handbook, so check each repository before committing to it.*

The techniques in this handbook reach most teams through a small number of open-source frameworks. This chapter maps the documented runs to the software they used, and summarizes the state of the main tools in each category.

### 21.1 What the documented runs used

| Run | Training software | Parallelism and precision |
|---|---|---|
| OPT-175B | metaseq: FSDP combined with Megatron-LM tensor parallelism | FP16 weights, FP32 Adam state sharded across hosts |
| BLOOM-176B | Megatron-DeepSpeed | TP=4, PP=12, DP=8, ZeRO stage 1, BF16 |
| Llama 3 405B | In-house | 4D parallelism (TP, CP, PP, DP with FSDP), BF16 |
| DeepSeek-V3 | HAI-LLM (in-house) | PP=16 with DualPipe, EP=64, ZeRO-1, no TP, FP8 |
| Kimi K2 | In-house | PP=16, EP=16, ZeRO-1, no TP, selective recomputation, FP8 storage for some activations, activation offload to CPU |
| OLMo 2 | OLMo (1B to 13B) and OLMo-core (32B): PyTorch with FSDP | BF16 |
| SmolLM3 | nanotron for training, datatrove for data, lighteval for evaluation, TRL for post-training | TP=2, DP=192, BF16 |
| Tulu 3 | open-instruct (SFT, DPO, RLVR) | |
| DAPO, and many open RL reproductions | verl | |
| GLM-4.5 | slime for RL | BF16 training, FP8 rollouts |

The frontier laboratories run in-house stacks. Everything that a team without one can adopt comes from the open projects in the remaining rows.

### 21.2 Pretraining frameworks

- **Megatron-LM and Megatron Core (NVIDIA).** The reference implementation of tensor, pipeline, data, expert, and context parallelism, with FP8 and FP4 support. NVIDIA now separates Megatron Core, a composable library, from Megatron-LM, the reference training scripts built on it, and provides Megatron-Bridge for converting checkpoints to and from the Hugging Face format. Actively developed. It is the usual choice for large dense or MoE pretraining on NVIDIA hardware.
- **NeMo (NVIDIA).** The NeMo repository changed scope in 2026 to speech, audio, and multimodal models, and its LLM collections were removed after release 2.7. Large-language-model training in NVIDIA's stack now lives in Megatron and in NeMo RL. Teams with NeMo-based LLM pipelines should plan a migration.
- **DeepSpeed (Microsoft).** ZeRO stages 1 to 3, offloading to CPU and NVMe, and sequence parallelism. Actively developed, and the basis of many other tools on this page.
- **PyTorch FSDP2 and torchtitan.** FSDP2 (the `fully_shard` API) shards each parameter with DTensor and replaces the original FSDP, which PyTorch's documentation now marks as deprecated. torchtitan is the PyTorch-native pretraining reference built on it, with tensor, pipeline, and context parallelism, Float8 and MXFP8 training, and `torch.compile`. Actively developed. It is the natural starting point for a team that wants to stay close to plain PyTorch.
- **nanotron (Hugging Face).** A compact 3D-parallel trainer used for SmolLM3 and documented in the Smol Training Playbook. The repository is maintained, although tagged releases are infrequent.
- **OLMo-core (Ai2).** The trainer behind OLMo 2 32B and OLMo 3, built on FSDP2 with `torch.compile`, asynchronous checkpointing, and official training scripts. Actively developed. Together with the OLMo papers, it is the most complete open example of a full pretraining stack.
- **GPT-NeoX (EleutherAI) and LLM Foundry with Composer (Databricks).** Both remain available and both see little recent activity. They are reasonable to read, and less reasonable to build on for a new project.

### 21.3 Fine-tuning toolkits

- **TRL (Hugging Face).** Trainers for SFT, DPO, GRPO, and reward modeling, integrated with PEFT, Accelerate, and DeepSpeed. It reached version 1.0 in March 2026 and is the default choice for post-training at small and medium scale.
- **PEFT (Hugging Face).** The standard implementation of LoRA and related methods (Ch. 19).
- **Axolotl.** Configuration-driven post-training (full fine-tuning, LoRA and QLoRA, preference methods, GRPO) on top of FSDP and DeepSpeed. Actively developed, and well suited to teams that prefer a YAML file to a training script.
- **Unsloth.** Optimized kernels for fine-tuning and RL on limited hardware, reporting about twice the speed and about 70% less memory than standard implementations. Actively developed. It is the usual choice for a single GPU.
- **LLaMA-Factory.** A unified interface for fine-tuning more than a hundred model families, with a command line and a web interface. Actively developed.
- **torchtune.** Development stopped in July 2025, and the repository now carries a notice that it is no longer actively maintained. New projects should not start on it.

### 21.4 RL post-training frameworks

- **verl.** The open implementation of ByteDance's HybridFlow, supporting PPO, GRPO, DAPO, and related algorithms, with FSDP or Megatron for training and vLLM or SGLang for rollouts. It is the most widely used open RL framework, and DAPO's released code is built on it.
- **OpenRLHF.** A framework based on Ray, vLLM, and DeepSpeed, with PPO, REINFORCE++, GRPO, and DPO. Actively developed.
- **open-instruct (Ai2).** The codebase behind Tulu 3 and the OLMo post-training recipes, including RLVR with GRPO. Its value is that the published recipes run as released.
- **NeMo RL (NVIDIA).** The successor to NeMo-Aligner, which was archived in November 2025. NVIDIA's Nemotron 3 Nano report names it as the RL framework.
- **AReaL, slime, prime-rl, SkyRL.** Four frameworks that emphasize asynchronous or agentic RL (Sec. 18.4): AReaL from Tsinghua and Ant Group, slime from Zhipu (used for GLM-4.5), prime-rl from Prime Intellect (used for the INTELLECT models), and SkyRL from UC Berkeley.

### 21.5 Data and evaluation tooling

- **datatrove (Hugging Face).** The pipeline library behind FineWeb: reading, filtering, deduplication, and tokenization, on a local machine, Slurm, or Ray. Actively developed.
- **NeMo Curator (NVIDIA).** GPU-accelerated curation, including exact, fuzzy, and semantic deduplication. Actively developed.
- **Dolma toolkit and DCLM.** The Ai2 toolkit behind the Dolma corpus, and the DataComp-LM testbed for controlled data experiments. Both remain available as references. Ai2's more recent data tools are separate projects.
- **lm-evaluation-harness (EleutherAI).** The most widely used evaluation harness, with many model back ends. Actively developed.
- **lighteval (Hugging Face) and OLMES (Ai2).** lighteval is the harness used for the FineWeb and SmolLM evaluations. OLMES implements the evaluation standard described in Sec. 20.1 and reproduces the OLMo and Tulu evaluations.

### 21.6 Kernels and low-precision libraries

- **FlashAttention.** FlashAttention-2 is the stable package. FlashAttention-3 (Hopper GPUs, with an FP8 forward pass) and a fourth generation for Hopper and Blackwell were both still in beta at the time of writing (Sec. 10.4).
- **Liger Kernel (LinkedIn).** Triton kernels for normalization, rotary embeddings, SwiGLU, and fused linear cross-entropy, which integrate with the Hugging Face trainer, TRL, Axolotl, and LLaMA-Factory.
- **Transformer Engine (NVIDIA).** The library that provides FP8 training on Hopper, Ada, and Blackwell GPUs, and MXFP8 and NVFP4 on Blackwell (Sec. 9.3). It is integrated into Megatron-LM, DeepSpeed, and Hugging Face Accelerate, among others.

### 21.7 Choosing a stack

| Situation | A reasonable default |
|---|---|
| One GPU, LoRA or QLoRA | Unsloth, or TRL with PEFT |
| One node, full fine-tuning and DPO | TRL or Axolotl with FSDP or DeepSpeed ZeRO-3, Liger kernels, FlashAttention |
| A few nodes, continued pretraining of a dense model up to about 30B | torchtitan or OLMo-core (FSDP2 with `torch.compile`), datatrove for data, lighteval or lm-evaluation-harness for evaluation |
| Large dense or MoE pretraining | Megatron-LM or Megatron Core, with Transformer Engine for FP8 |
| RLVR or GRPO | verl or open-instruct. TRL is sufficient at small scale |
| Agentic RL with long rollouts | A framework with an asynchronous mode (verl, AReaL, slime, prime-rl) |

Two selection rules matter more than any feature list. Prefer a tool in which a published recipe close to the intended run is known to work as released, because reproducing a documented result is the best acceptance test of an installation. Check activity before adopting: the record of 2025-2026 includes one widely used library that stopped development (torchtune), one that was archived (NeMo-Aligner), and one that changed scope (NeMo).

**Checkpoint question.** A team is about to start a 20B-token continued-pretraining run of an 8B model on two 8×H100 nodes. Which components does it need, and what should it verify before the real run?

*Answer.* Components: a trainer with FSDP2 or ZeRO-3 sharding and BF16 (torchtitan, OLMo-core, or a Hugging Face stack), FlashAttention and fused cross-entropy, a pre-tokenized and packed dataset with document masking, local NVMe for data and checkpoints, and an evaluation harness running on a schedule. Verification before the real run: (1) a determinism test, in which the same seed gives the same loss for 100 steps under the chosen parallelism (Sec. 10.3); (2) a throughput test, which should show MFU of 30% or better (Sec. 10.4); (3) a checkpoint save, a simulated failure, and a resume, with the loss curve continuing exactly; (4) an evaluation of the *unmodified* base model through the harness, to confirm that the scores match published values before any change is attributed to training; (5) a short run on 1% of the data with the full schedule, including the decay phase, to exercise the entire pipeline.

**References:**
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM · Megatron-Bridge: https://github.com/NVIDIA-NeMo/Megatron-Bridge
- NeMo: https://github.com/NVIDIA-NeMo/NeMo · NeMo RL: https://github.com/NVIDIA-NeMo/RL
- DeepSpeed: https://github.com/deepspeedai/DeepSpeed
- PyTorch FSDP2 tutorial: https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- torchtitan: https://github.com/pytorch/torchtitan and paper: https://arxiv.org/abs/2410.06511
- nanotron: https://github.com/huggingface/nanotron
- OLMo-core: https://github.com/allenai/OLMo-core
- GPT-NeoX: https://github.com/EleutherAI/gpt-neox · LLM Foundry: https://github.com/mosaicml/llm-foundry
- TRL: https://github.com/huggingface/trl · PEFT: https://github.com/huggingface/peft
- Axolotl: https://github.com/axolotl-ai-cloud/axolotl · Unsloth: https://github.com/unslothai/unsloth · LLaMA-Factory: https://github.com/hiyouga/LlamaFactory
- torchtune, notice of end of development: https://github.com/meta-pytorch/torchtune/issues/2883
- verl: https://github.com/verl-project/verl and HybridFlow: https://arxiv.org/abs/2409.19256
- OpenRLHF: https://github.com/OpenRLHF/OpenRLHF · open-instruct: https://github.com/allenai/open-instruct
- AReaL: https://github.com/areal-project/AReaL · slime: https://github.com/THUDM/slime · prime-rl: https://github.com/PrimeIntellect-ai/prime-rl · SkyRL: https://github.com/NovaSky-AI/SkyRL
- datatrove: https://github.com/huggingface/datatrove · NeMo Curator: https://github.com/NVIDIA-NeMo/Curator · Dolma toolkit: https://github.com/allenai/dolma · DCLM: https://github.com/mlfoundations/dclm
- lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness · lighteval: https://github.com/huggingface/lighteval · OLMES: https://github.com/allenai/olmes
- FlashAttention: https://github.com/Dao-AILab/flash-attention · Liger Kernel: https://github.com/linkedin/Liger-Kernel · Transformer Engine: https://github.com/NVIDIA/TransformerEngine

## 22. Incident Index and Pre-Flight Checklists

*Applies to: all readers. The index is a reference, and the checklists are meant to be copied into a project's own documentation.*

### 22.1 Incident index

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
| Frontier reasoning model in agentic coding RL (OpenAI, 2025) | The policy learned to exit before tests ran, to skip tests, and to read expected values from test files | Verifiers that could be satisfied without solving the task | Hardened verifiers; a chain-of-thought monitor used for diagnosis and not as a training signal (Ch. 18) |
| SWE-bench Verified (2025-2026) | Agents retrieved fixes from future repository history; the benchmark was later retired by its sponsor | Task repositories that contained the solution; test and description flaws; training-data exposure | Repositories stripped of remotes, branches, tags, and reflogs; no network access; private and rotating evaluations (Ch. 18, 20) |

### 22.2 Pre-flight checklist: pretraining and continued pretraining

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

### 22.3 Pre-flight checklist: post-training

- [ ] Capability targets enumerated; the evaluation suite built first, including evaluations of feared failures (sycophancy, refusal drift, length inflation, dialect drift)
- [ ] SFT data decontaminated; response-only masking verified on a decoded batch; chat template frozen
- [ ] Loss normalization verified across gradient accumulation and data-parallel ranks (Sec. 14.2)
- [ ] Preference stage: on-policy pairs in the mixture; a check for length bias; reward composition reviewed as a design document
- [ ] RLVR: verifiers tested adversarially; generation and training throughput budgeted; KL and entropy monitors enabled
- [ ] Agentic RL: tool outputs masked from the loss; sandboxes without network access or repository history; high-reward trajectories read by a person at intervals (Ch. 18)
- [ ] Distillation: the teacher's license checked; teacher outputs filtered for correctness and for language (Ch. 17)
- [ ] Qualitative transcript review scheduled, with blocking authority
- [ ] Rollback plan: the previous checkpoint deployable in minutes. The central lesson of the sycophancy incident is that the best-resourced laboratory needed one

**References:** the sources for each incident are listed in the chapter named in the last column of the index.

## 23. Sources and Further Reading

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
- On-Policy Distillation (Thinking Machines): https://thinkingmachines.ai/blog/on-policy-distillation/
- Distillation Scaling Laws: https://arxiv.org/abs/2502.08606
- DeepSeek-V3.2: https://arxiv.org/abs/2512.02556 and GLM-4.5: https://arxiv.org/abs/2508.06471
- Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation: https://arxiv.org/abs/2503.11926

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
- ArabicMMLU: https://arxiv.org/abs/2402.12840, the Open Arabic LLM Leaderboard v2: https://huggingface.co/blog/leaderboard-arabic-v2, and BALSAM: https://arxiv.org/abs/2507.22603

---

*Descriptions and figures were compiled from the cited reports as of September 2026. Corrections are welcome: if a figure has drifted or a claim is wrong, open an issue or a pull request.*

**Related repositories:** [LLM-Inference-Handbook](https://github.com/h9-tec/LLM-Inference-Handbook) · [LLM-Math-Handbook](https://github.com/h9-tec/LLM-Math-Handbook) · [llm-systems-engineering-roadmap](https://github.com/h9-tec/llm-systems-engineering-roadmap) · [Awesome_Arabic_NLP](https://github.com/h9-tec/Awesome_Arabic_NLP)
