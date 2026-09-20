# LLM Training Handbook

**How large language models actually get trained: the mechanisms, the economics, and what really happens when thousands of GPUs run for weeks.**

Most training tutorials teach you the loss function and stop there. This handbook is built the other way around: every chapter is anchored in **documented, real training runs**, the OPT-175B logbook, the Llama 3.1 405B reliability report, DeepSeek-V3's FP8 recipe, Kimi K2's zero-spike run, OLMo 2's stability investigation, Hugging Face's SmolLM3 restart-after-1T-tokens story, the Tulu 3 and DeepSeek-R1 post-training pipelines, and the Arabic adaptation literature (Jais, AceGPT, ALLaM, Kuwain). The goal is that after reading it, a training log, a technical report, or a divergence at 3am all look familiar to you.

This is the training-side companion to the [LLM-Inference-Handbook](https://github.com/h9-tec/LLM-Inference-Handbook) and the [LLM-Math-Handbook](https://github.com/h9-tec/LLM-Math-Handbook). Inference asks "how do we serve this model cheaply?"; this book asks "how did this model come to exist, and what broke along the way?"

**Who it's for:** ML engineers, infra engineers, and technical leads who train, fine-tune, or adapt LLMs, or who need to read training reports critically. Interview candidates for training and infrastructure roles will find the standard questions covered with real numbers.

**You need:** nothing to install. Comfort with basic ML and transformer concepts helps. Arabic-specific sections assume no prior Arabic NLP knowledge.

---

## Contents

**Part I: The map**

1. [What "training" means in 2026: the full pipeline](#1-what-training-means-in-2026-the-full-pipeline)
2. [The economics: what training actually costs](#2-the-economics-what-training-actually-costs)

**Part II: Pretraining**

3. [Data engineering: the FineWeb school](#3-data-engineering-the-fineweb-school)
4. [Tokenizers, and the Arabic tokenization tax](#4-tokenizers-and-the-arabic-tokenization-tax)
5. [Scaling laws, budgets, and the ablation game](#5-scaling-laws-budgets-and-the-ablation-game)
6. [Architecture choices that are really training choices](#6-architecture-choices-that-are-really-training-choices)
7. [Optimizers, schedules, and hyperparameters that matter](#7-optimizers-schedules-and-hyperparameters-that-matter)
8. [Stability: the physics of loss spikes](#8-stability-the-physics-of-loss-spikes)
9. [Numerics: FP16 pain, BF16 peace, FP8 frontier](#9-numerics-fp16-pain-bf16-peace-fp8-frontier)
10. [Distributed training: the parallelism zoo and real MFU](#10-distributed-training-the-parallelism-zoo-and-real-mfu)
11. [Hardware reliability: a failure every three hours](#11-hardware-reliability-a-failure-every-three-hours)

**Part III: Mid-training and adaptation**

12. [Mid-training: annealing, curricula, and long context](#12-mid-training-annealing-curricula-and-long-context)
13. [Continued pretraining and new-language adaptation: the Arabic deep dive](#13-continued-pretraining-and-new-language-adaptation-the-arabic-deep-dive)

**Part IV: Post-training**

14. [SFT that actually works](#14-sft-that-actually-works)
15. [Preference optimization, and the sycophancy incident](#15-preference-optimization-and-the-sycophancy-incident)
16. [RL for reasoning: GRPO, RLVR, and the R1 pipeline](#16-rl-for-reasoning-grpo-rlvr-and-the-r1-pipeline)

**Part V: The practitioner's track**

17. [Fine-tuning without a cluster: LoRA, full FT, and the decision tree](#17-fine-tuning-without-a-cluster-lora-full-ft-and-the-decision-tree)
18. [Evaluation during training](#18-evaluation-during-training)
19. [The war-stories index and pre-flight checklists](#19-the-war-stories-index-and-pre-flight-checklists)
20. [Sources and further reading](#20-sources-and-further-reading)

---

# Part I: The map

## 1. What "training" means in 2026: the full pipeline

When a lab says "we trained a model," they mean a pipeline with four to six distinct phases, each with its own data, hyperparameters, failure modes, and team. Collapsing them into one word is the single most common source of confusion when reading technical reports.

```
raw web + curated sources
        |
        v
[ Data engineering ]  -> filtering, dedup, classification, mixing   (Ch. 3)
        |
        v
[ Pretraining ]       -> next-token prediction on trillions of tokens (Ch. 5-11)
        |
        v
[ Mid-training ]      -> annealing on high-quality mixes, curriculum
                         shifts, long-context extension               (Ch. 12)
        |
        v
[ Post-training ]
    SFT               -> imitate curated demonstrations               (Ch. 14)
    Preference (DPO/RLHF) -> optimize toward human/AI preferences     (Ch. 15)
    RL / RLVR         -> optimize verifiable or judged outcomes       (Ch. 16)
        |
        v
released model  (base + instruct/thinking variants)
```

**The vocabulary, precisely:**

- **Pretraining** is self-supervised next-token prediction on a huge corpus. The objective for a sequence of tokens $x_1..x_T$ is the cross-entropy $\mathcal{L} = -\sum_t \log p_\theta(x_t \mid x_{<t})$. It consumes 95%+ of the FLOPs and produces a *base model*: a text-completion engine with broad knowledge but no conversational behavior.
- **Mid-training** (also called *annealing* or *stage-2/3 pretraining*) is the modern refinement: the last portion of pretraining is run on a deliberately upgraded data mixture (more STEM, code, math, long documents) while the learning rate decays. Qwen3 formalizes this as three explicit stages: over 30T general tokens at 4K context, then roughly 5T knowledge-intensive tokens, then a long-context stage extending to 32K. OLMo 2 similarly restarts from the pretrained checkpoint on domain-specific mixtures with the learning rate driven linearly to zero. This stage is where a lot of benchmark performance is actually won, cheaply.
- **Long-context extension** usually happens here too: a short additional run on longer sequences, often with RoPE rescaling tricks like YaRN. Kimi K2 trained 400B annealing tokens at 4K and only 60B at 32K, then used YaRN to reach 128K. Long context is bought with a tiny fraction of total tokens.
- **SFT (supervised fine-tuning)** teaches the base model the *format and behavior* of an assistant by imitating curated prompt-response pairs, with loss masked to the response tokens.
- **Preference optimization** (RLHF with PPO, or the direct methods: DPO and its variants) pushes the model toward outputs humans (or AI judges) prefer between alternatives.
- **RL with verifiable rewards (RLVR)** and reasoning RL (GRPO and friends) optimize the model against programmatic checkers (math answers, unit tests, format validators) rather than learned reward models. This is the engine behind the reasoning-model wave started by DeepSeek-R1.

**Who actually does which phase.** Very few organizations pretrain from scratch; the honest first question in Hugging Face's Smol Training Playbook is whether you should train at all, given how strong open bases (Qwen, Llama, Gemma, DeepSeek) already are. The realistic map:

| You are | Phases you'll actually run |
|---|---|
| Frontier lab / national program (SDAIA, G42, Moonshot) | All of them, from raw data to RL |
| Serious regional player with a GPU cluster | Continued pretraining + full post-training on an open base (the ALLaM / AceGPT path, Ch. 13) |
| Product team with modest GPUs | SFT + DPO, maybe LoRA-based, on an instruct or base model (Ch. 14-17) |
| Most teams | No training at all: prompting + RAG, revisited quarterly as open models improve |

Parts II and III mostly concern the first two rows, Parts IV and V the third. Readers in the fourth row should read Ch. 17 and Ch. 18 before deciding to train at all.

## 2. The economics: what training actually costs

### 2.1 The compute arithmetic

The standard estimate for training compute is:

$$C \approx 6 \cdot N \cdot D \quad \text{FLOPs}$$

where $N$ is parameter count and $D$ is training tokens. The 6 comes from roughly 2 FLOPs per parameter per token for the forward pass and 4 for the backward pass. For MoE models, $N$ is the *activated* parameter count per token, which is exactly why sparse models changed the economics: DeepSeek-V3 has 671B total parameters but only 37B active, so each token costs compute like a 37B dense model while the model stores knowledge like a much larger one.

Convert FLOPs to time with:

$$\text{days} = \frac{6ND}{\;\text{GPUs} \times \text{peak FLOPs/s} \times \text{MFU} \times 86{,}400\;}$$

**MFU (Model FLOPs Utilization)** is the fraction of theoretical peak your training actually achieves, and it is where all the engineering lives. Real, published values, worth memorizing as calibration points:

| Run | Hardware | MFU / throughput | Source |
|---|---|---|---|
| OPT-175B (2021) | 992× A100 80GB | up to 147 TFLOP/s per GPU (~47% of A100 BF16 peak) | OPT paper |
| BLOOM-176B (2022) | 384× A100 80GB | ~156 TFLOP/s per GPU (~50%) | GENCI / BigScience |
| MegaScale 175B (2024) | 12,288 GPUs | 55.2% MFU (1.34× over stock Megatron-LM) | ByteDance NSDI'24 |
| Llama 3.1 405B (2024) | 16,384× H100 | ~38-43% (380-430 TFLOPs/GPU, BF16) | Llama 3 paper |
| SmolLM3 3B (2025) | 384× H100 | ~30% end-to-end MFU (kernel-level matmuls hit 72-77% of peak; communication and non-matmul ops eat the rest) | Smol Training Playbook |

The gap between "the GPU's spec sheet" and "your training run" is routinely 2-3×. When someone quotes you a training timeline computed at peak FLOPs, multiply by 2.5 and you will be closer to reality.

### 2.2 Real invoices

**DeepSeek-V3, the efficiency benchmark.** The technical report breaks the bill down precisely: 2.664M H800 GPU-hours for pretraining 14.8T tokens (about 180K GPU-hours per trillion tokens, i.e. 3.7 days per trillion on their 2,048-GPU cluster), plus 119K GPU-hours for context extension and only 5K for post-training, totaling 2.788M GPU-hours. At an assumed $2/GPU-hour rental that is the famous **$5.576M**. The number is accurate for what it measures and misleading as a comparison:

- Honest: it was achieved with architecture-training co-design (MoE with 37B active, MLA to shrink memory, FP8 compute, multi-token prediction densifying each step, custom DualPipe communication to work around the H800's deliberately weakened interconnect). The report also states the run had **no irrecoverable loss spikes and no rollbacks**, which itself saves enormous money (see Ch. 8).
- Misleading: the figure covers only the *final* run. It excludes every ablation, failed experiment, researcher salary, and the capital cost of owning 2,048 H800s (far more than $5.5M). Comparing it against another lab's fully-loaded budget is an apples-to-invoices error that half of LinkedIn made in January 2025.

**Llama 3.1 405B** used 30.84M GPU-hours on H100s for a similar token count (~15T), roughly 11× DeepSeek-V3's hours. Dense architecture, BF16 everywhere, and a much larger active parameter count per token explain most of the gap: this is the cost of not co-designing for efficiency (and of training a year earlier).

**BLOOM-176B** cost about 1.08M A100-hours over 3.5 months on France's Jean Zay supercomputer (estimated $2-5M cloud-equivalent including preliminary experiments), consuming 433 MWh. Useful as the "public-sector, 2022 technology" reference point.

**SmolLM3-3B** is the best-documented small run. The Smol Training Playbook publishes GPU-hours, not dollars: 276,480 H100-hours for the main 11T-token run (384 H100s for about a month), plus 161,280 H100-hours of pretraining ablations, mid-training ablations, and a restart with its debugging, for 437,760 H100-hours in total. At an assumed $2-3 per H100-hour that is roughly $0.9-1.3M; the dollar figure is an estimate derived here, not one published by Hugging Face. More than a third of the total (37%) was spent outside the final run. The expensive part of training is rarely the planned run; it is the ablations and the restarts, and Ch. 19 catalogs the documented ones.

### 2.3 The hidden line items

Budgets that only count the final run miss, in rough order of size:

1. **Ablations and dead ends.** SmolLM3's team trained the full 3B architecture on 100B-token slices repeatedly to test data and architecture choices; Moonshot ran Kimi K2 ablations on a 3B-total/0.5B-active proxy MoE because full-size ablations at 1T parameters were unaffordable. A serious run budgets 10-30% of final-run compute for ablations, and skipping this is how you discover a bad decision 1T tokens in.
2. **Restarts and instability.** PaLM's team handled around 20 loss spikes by rewinding and skipping batches; the ZClip paper, citing LLM360's K2-65B run, puts the cost of handling that run's loss spikes at an additional 30 days and 129.3 MWh; SmolLM3 restarted from scratch after burning 1T tokens on a seed bug. Effective training time on Llama 3.1 405B was above 90%, which is considered excellent, and still means paying for 16,384 H100s during the other ~10%.
3. **Storage and I/O.** A full BLOOM checkpoint (weights + optimizer states) is 2.3TB, about 13 bytes per parameter; you keep many of them, and you must be able to write them fast enough that checkpointing doesn't stall 384+ GPUs (Ch. 10-11).
4. **People.** An on-call rotation for a multi-week run is a real staffing cost; the OPT logbook is, among other things, a diary of engineers being paged through Christmas.

**Interview checkpoint (Ch. 2):** Given a 70B dense model, 15T tokens, 1,024 H100s (989 TFLOPs BF16 peak), and 40% MFU, estimate wall-clock. $C = 6 \times 70\text{e}9 \times 15\text{e}12 = 6.3\text{e}24$ FLOPs. Cluster delivers $1024 \times 989\text{e}12 \times 0.4 = 4.05\text{e}17$ FLOP/s. That is $1.56\text{e}7$ s ≈ **180 days**, before failures. This one calculation, done in your head, filters most vendor claims.

---

# Part II: Pretraining

## 3. Data engineering: the FineWeb school

Data work is 60% of a pretraining project's calendar time and gets 5% of the attention in tutorials. The FineWeb project (Hugging Face, 2024) changed that by publishing not just a 15-trillion-token dataset from 96 Common Crawl snapshots, but the full experimental record of how every curation decision was validated. Treat it as the reference methodology.

### 3.1 The pipeline, stage by stage

**Extraction.** Raw Common Crawl WARC files contain full HTML. FineWeb extracts main-content text with trafilatura, chosen empirically: models trained on trafilatura-extracted text beat models trained on the default WET extractions on downstream benchmarks, because boilerplate (menus, cookie banners, footers) is training poison. Lesson one: your HTML extractor is a model-quality decision, not a plumbing detail. (For Arabic, extraction is harder still: encoding issues, mixed RTL/LTR markup, and diacritics stripping by naive pipelines.)

**Language identification.** A fastText-style classifier with a threshold (FineWeb keeps documents with P(English) ≥ 0.65). For Arabic corpora the equivalent step must additionally decide what to do with dialects, romanized Arabic (Arabizi), and heavy code-switching, which off-the-shelf LID handles poorly; ALLaM's team built their own filtering pipeline for a 540B-token Arabic corpus, half of it machine-translated from English (Ch. 13).

**Quality filtering.** FineWeb layers MassiveText-style and C4-style heuristics with a small set of custom filters, and, critically, **ablated each filter** instead of trusting intuition. For the custom filters the team computed more than fifty document statistics, derived seventeen candidate metric-threshold pairs from them, and kept only the three that survived ablation (fraction of lines ending in punctuation, fraction of characters in duplicated lines, fraction of lines shorter than 30 characters), which together remove about 22% of tokens. C4's rule dropping lines without terminal punctuation gave the largest single gain of any C4 filter but removed about 30% of all tokens, so it was dropped; the remaining C4 filters together did better while removing about 7%. Lesson two: every filter is a hypothesis; the only proof is training a model with and without it.

**Deduplication, the counterintuitive result.** The team originally planned global MinHash deduplication across all 96 snapshots. Ablations showed the opposite of intuition: **per-snapshot dedup outperformed global dedup**. Global dedup preferentially deleted older, higher-quality content that recurs across years (the same good article crawled 20 times) while what survived globally skewed worse; deduplicating each crawl independently kept quality up. Their production config: MinHash over 5-grams, 112 hash functions split into 14 buckets of 8, roughly a 75% similarity threshold. A second surprise: after the FineWeb-Edu educational filter, additional dedup produced no measurable gain in their 1.8B/350B-token ablation. Deduplication interacts with everything else; measure, don't assume. (The classic Lee et al. result still holds as the baseline motivation: dedup cuts verbatim memorization by an order of magnitude and reaches the same loss in fewer steps.)

**Model-based quality classification.** FineWeb-Edu filtered the 15T corpus down to 1.3T tokens using a small classifier trained on Llama-3-70B judgments of "educational value", and the filtered subset **outperforms the full dataset** on knowledge benchmarks. This is the single most copied idea of 2024-2026: nearly every serious lab now runs learned quality/domain classifiers over the whole corpus. Qwen3 pushed it furthest, labeling a 30T+ token corpus instance-by-instance along dimensions like educational value, domain, and safety, then optimizing the mixture at the instance level via small-proxy ablations.

**Synthetic data joins the mixture.** Qwen3's corpus (36T tokens, 119 languages) was partly manufactured by its own ancestors: Qwen2.5-VL extracted text from PDF-like documents, Qwen2.5 cleaned it, and Qwen2.5-Math/Qwen2.5-Coder generated textbooks, QA pairs, and code. The earlier generation of a model family has become a data factory for the next. The risks (distribution narrowing, error amplification, benchmark leakage through synthetic channels) are managed the same way as everything else in this chapter: ablate on proxies, decontaminate against your eval suite (Ch. 18).

**Data-side stability hygiene.** OLMo 2 filtered documents containing long repeated n-grams after tracing loss spikes partly to them: a page repeating the same token pattern thousands of times produces gradient pathologies (Ch. 8). Data cleaning is also a stability intervention.

### 3.2 Mixing: the ratios are the recipe

Given cleaned sources (web, code, math, papers, books, multilingual), the mixture weights are among the most consequential hyperparameters in the whole project, and the modern method for setting them is empirical: train small proxies on candidate mixtures, evaluate on early-signal benchmarks, extrapolate. OLMo 2 introduced "microannealing" for exactly this: short, cheap decay-phase runs on a candidate data source mixed into a base mixture, to measure that source's marginal value before buying it a seat in the real run. Llama 3's team similarly used small annealing runs to score new data sources. The general findings that replicate across reports:

- Code in the mixture helps general reasoning, not just coding.
- Upsampling a high-quality source works up to a few epochs, then decays; Jais upsampled its 72B unique Arabic tokens to ~116B effective (about 1.6 epochs) because Arabic supply was the binding constraint (Ch. 13).
- The best data is saved for last: quality-weighted curricula (plain web early, textbook-grade and reasoning-heavy late, during LR decay) consistently beat uniform mixing. This is the entire logic of mid-training (Ch. 12).

### 3.3 What this means if you are building an Arabic corpus

Everything above, plus: the public Arabic web is a fraction of English (Jais's 72B-token corpus was the largest Arabic collection of its time, versus multi-trillion-token English corpora); Common Crawl's Arabic slice is disproportionately boilerplate, mirrored news, and religious text (fine content, but a skewed distribution); dialectal text lives on social platforms with restrictive terms; and OCR of Arabic books remains a genuine data moat for whoever does it well (layout, ligatures, diacritics). This scarcity is why every Arabic model story in Ch. 13 is fundamentally a data story, and why translated data (ALLaM used translated corpora deliberately, and roughly a quarter of Jais's Arabic was translated from English) shows up despite its known style artifacts: the alternative was not having the tokens at all.

## 4. Tokenizers, and the Arabic tokenization tax

The tokenizer is fixed before training and haunts every stage after it. It determines (a) how much text fits in a context window, (b) what a "token" of compute buys you per language, and (c) how hard adaptation to a new language will be.

### 4.1 Mechanics in one paragraph

Byte-level BPE starts from bytes and greedily merges the most frequent adjacent pairs until reaching a target vocabulary size, on a tokenizer-training corpus whose language balance silently sets each language's efficiency. The key measured quantity is **fertility**: average tokens per word. A tokenizer trained mostly on English will shatter Arabic words into many pieces, and Arabic's templatic morphology (root + pattern + clitics: و+سيكتبون+ها as one orthographic word) makes it especially vulnerable.

### 4.2 The tax, in numbers that hit the budget

High fertility on your language means, simultaneously: fewer effective words per context window, more tokens (thus more compute and more money) to represent the same corpus, higher serving cost per answer, and worse per-token learning signal. The published measurements are large. Gosal et al. report Llama-2's tokenizer at 5.06 tokens per Arabic word, against 1.41 after adding 32K Arabic tokens, and ALLaM charts the same effect for its merged tokenizer. At those figures the same Arabic dataset costs about 3.6 times as many tokens to train through with the English-centric tokenizer. When you read "trained on X trillion tokens," always ask: whose tokens?

Two engineering responses exist:

1. **Train the tokenizer on the right balance from the start** (the from-scratch path). Jais trained a balanced bilingual vocabulary; BLOOM trained a 250,680-entry multilingual vocabulary with per-language alpha-weighting.
2. **Vocabulary expansion of an existing model** (the adaptation path): merge new-language tokens into a pretrained model's tokenizer, extend the embedding matrix, and initialize each new token's embedding as the average of the old-tokenizer subtokens that compose it (subtoken-mean initialization, the method formalized as Fast Vocabulary Transfer by Gee et al., 2022, and used by ALLaM and RightNow-Arabic; WECHSEL is a different method that relies on aligned bilingual word embeddings). This is now the standard first move of language adaptation and is covered operationally in Ch. 13.

3. A research third way: morphology-aware tokenization, like the MorphBPE line from the Fanar project, which respects Arabic morpheme boundaries inside BPE. (See [MorphBPE](https://github.com/h9-tec/MorphBPE) for an implementation.)

### 4.3 Practical checks before you commit a tokenizer

- Measure fertility on *your* target distributions (MSA news, dialectal chat, code-switched support tickets), not on a generic corpus. A model for Saudi call centers lives on dialect, and fertility there is usually worse than on MSA.
- Check digit handling (individual digits vs. chunks), whitespace and newline treatment (matters for code), diacritics behavior (are they separate tokens? stripped?), and Arabic presentation forms / Unicode normalization (NFKC or not, decide once).
- Vocabulary size trades embedding-parameter cost against sequence length; at small model scale the embedding matrix can dominate parameters (a 0.5B model with a 150K vocabulary spends a huge fraction of its weights on embeddings), which is why small-model adapters inject only the most valuable new tokens (Kuwain added Arabic tokens to TinyLlama; RightNow injected 27,032 Arabic tokens into Qwen2.5-0.5B rather than retraining the whole vocabulary).

## 5. Scaling laws, budgets, and the ablation game

### 5.1 Chinchilla and what replaced it in practice

The Chinchilla result says that for a fixed compute budget $C = 6ND$, loss is minimized around $D \approx 20N$: a 10B model wants ~200B tokens. Every modern open model violates this on purpose, in the overtrained direction: Llama 3 8B saw ~15T tokens (about 1,875 tokens per parameter), Qwen3's dense models trained on 36T, SmolLM3-3B on 11T. The reason is that Chinchilla optimizes *training* compute only, while the thing being minimized commercially is *training + lifetime inference* cost. A smaller model trained far past compute-optimality is cheaper to serve forever, so the industry systematically buys extra pretraining tokens to shrink the deployed model. When you see a "compute-optimal" claim, ask which cost function.

The other quiet revolution: scaling laws stopped being just about N and D and became a hyperparameter-transfer tool. Qwen3 reports scaling-law-guided tuning of learning rate schedules and batch sizes, fit separately for dense and MoE models and per training stage. The pattern: fit trends on a ladder of small models, predict the large model's optimal settings, verify with one or two spot checks.

### 5.2 The ablation game: how decisions actually get made

No lab decides architecture or data questions by argument; they decide by proxy runs, and the craft is in making proxies predictive:

- **Same size, fewer tokens.** SmolLM3 ran its ablations as the full 3B architecture on 100B tokens (roughly 1% of the final run). For readers, the playbook reproduces the same comparisons on a 1B model trained on 45B tokens, about a day and a half on one 8×H100 node. Cheap enough to iterate, big enough to transfer.
- **Smaller proxy of the same shape.** Kimi K2 (1T total / 32B active) ran ablations on a 3B-total / 0.5B-active MoE with matched design ratios. Vanilla-Muon instability was caught on a 53B-total / 9B-active mid-scale run before it could destroy the real one (Ch. 8): the proxy ladder is also your safety net.
- **Early-signal evaluation.** At ablation scale, most benchmarks are noise. The Smol playbook's answer is task formulation: cloze-style likelihood scoring gives usable signal on small models where multiple-choice and free-generation formats are still flat (more in Ch. 18).
- **Decay-phase probes.** OLMo 2's microannealing (Sec. 3.2) turns the LR-decay phase into a measurement instrument for data value.

The discipline to internalize: **every opinion about training is either an ablation result or a guess.** The published reports that read as confident (Qwen3's stage ratios, FineWeb's filter set, OLMo 2's stability package) are compressed summaries of hundreds of small runs. Budget for yours (10-30% of final-run compute, Sec. 2.3), and log them so the next project inherits the answers.

### 5.3 Small-scale corollary

The same methodology scales down honestly. If your "final run" is a 1.5B Arabic adaptation on 8×H100 (Ch. 13's Kuwain/RightNow tier), your ablation tier is a 0.3-0.5B model on a few billion tokens, and the questions are identical: which vocabulary injection size, which replay ratio of original-language data, which LR. Teams that skip straight to the final run at this scale lose more GPU-days to re-runs than the ablations would have cost.

## 6. Architecture choices that are really training choices

Architecture papers frame their choices as modeling ideas. Reading the 2024-2026 reports closely, most load-bearing choices are actually about training economics and stability.

**MoE is a cost decision.** A mixture-of-experts layer routes each token to a few experts out of many, decoupling knowledge capacity (total parameters) from per-token compute (active parameters). DeepSeek-V3: 671B total, 37B active. Kimi K2: 1T total, 32B active across 384 experts. The training bill scales with active parameters (Sec. 2.1), which is how a $5.6M final run and a 1T-parameter model coexist. The price is paid in engineering: expert parallelism, all-to-all communication, and the load-balancing problem: if routing collapses onto a few experts, you paid for parameters that never train. The classical fix is an auxiliary load-balancing loss, which distorts the main objective; DeepSeek-V3's auxiliary-loss-free strategy (bias-based routing adjustment) and Qwen3's global-batch load-balancing loss are two different resolutions of the same tension. If you evaluate MoE claims, ask three questions: active params per token, expert-parallel communication cost on the actual interconnect, and what keeps routing balanced.

**MLA is a memory decision.** DeepSeek's Multi-head Latent Attention compresses keys/values into a low-rank latent, cutting KV-cache size massively. It is presented as an inference feature, but during training it also shrinks activation memory and communication, part of why V3's hours are so low. GQA (grouped-query attention: many query heads share fewer KV heads) is the mainstream version of the same trade; SmolLM3 uses GQA with 4 groups, Qwen3-8B runs 32 query heads against 8 KV heads.

**Stability features are now architecture.** QK-norm (RMSNorm applied to queries and keys before the attention dot product) appears in OLMo 2, Qwen3 (all models), Gemma, and many 2025-2026 reports, because unbounded attention logits are a leading cause of divergence (Ch. 8). OLMo 2 also reordered normalization (normalizing sublayer *outputs*) and adopted z-loss on the final softmax. OLMoE measured QK-norm costing almost 10% throughput and kept it anyway; stability is worth paying for. When you see these in a model card, read them as scar tissue from someone's diverged run.

**Multi-token prediction (MTP) is a signal-density decision.** DeepSeek-V3 adds a sequential prediction module so that each position also predicts one additional future token (prediction depth 1), extracting more gradient signal per sequence (and doubling as a speculative-decoding module at inference). More learning per token processed = fewer GPU-hours per unit of capability.

**Positional-encoding choices are context-cost decisions.** RoPE everywhere, with the modern refinements aimed at cheap long-context extension (Ch. 12): partial/adjusted RoPE, NoPE-hybrid layers (SmolLM3), and YaRN-style rescaling at extension time (Kimi K2 to 128K). ALiBi (BLOOM) was an earlier attempt at the same goal.

**The meta-lesson.** Between 2022's BLOOM/OPT and 2026's frontier, the transformer block barely changed; what changed is that every remaining choice is justified by ablations against a cost function that includes training stability and serving cost. Architecture reviews in your team should be run the same way: not "is this idea elegant" but "what does it do to tokens/sec, memory, spike risk, and the inference bill."

## 7. Optimizers, schedules, and hyperparameters that matter

### 7.1 AdamW, and the settings people get wrong

AdamW remains the default: momentum ($\beta_1 \approx 0.9$), a second-moment normalizer ($\beta_2 \approx 0.95$ for LLMs, not the 0.999 default from vision), decoupled weight decay (~0.1), gradient clipping at norm 1.0. Three empirically load-bearing details from real reports:

- **Epsilon matters.** OLMo 2 moved Adam's $\epsilon$ from 1e-5 to 1e-8 and measured faster early loss improvement and a lower, more stable gradient norm. A too-large $\epsilon$ quietly damps the update exactly when gradients are small.
- **Do not weight-decay the embeddings.** OLMo 2 traced part of its instability to decayed embeddings shrinking too far; small embedding norms inflate early-layer gradients (the LayerNorm Jacobian scales like $1/\lVert x \rVert$) and produced measurably more spikes. Exempt embeddings (and norms) from decay. (OLMoE found the effect minor at its scale and kept weight decay on all parameters, a reminder that these choices interact with size; when in doubt, follow the larger-scale evidence.)
- **FP32 optimizer state, sharded.** Even in mixed precision, Adam's moments and the master weights stay in FP32; OPT kept Adam state in FP32 by sharding it across hosts while model weights ran FP16. This is why optimizer memory is ~8-12 bytes/param and why ZeRO/FSDP sharding exists (Ch. 10).

### 7.2 The Muon storyline: a 2025 changing of the guard

Muon is a matrix-aware optimizer (momentum with a spectral orthogonalization step) that reaches a given loss in noticeably fewer tokens than AdamW. Moonshot bet Kimi K2 on it and hit Muon's known failure mode at scale: on a 53B-total/9B-active proxy, **maximum attention logits blew past 1,000**, the precursor of spikes and divergence. Their fix, MuonClip, wraps Muon with weight decay, RMS-matched update scale, and **QK-Clip**: monitor each attention head's max logit, and when it exceeds a threshold $\tau$, rescale that head's query/key projection weights directly. The result is the most striking stability datapoint in the public record: **15.5T tokens at 1T parameters with zero loss spikes**, a per-step loss curve published unsmoothed. Two portable lessons even if you never touch Muon: (1) max attention logit is a first-class monitoring signal, and (2) intervening on *weights* (rescaling) is stronger medicine than intervening on *gradients* (clipping), because it removes the cause rather than the symptom.

### 7.3 Learning-rate schedules: cosine vs. WSD

- **Warmup** (linear, 500-2000 steps in most reports: OLMo 2 uses 2000, K2 used 500) exists because Adam's second-moment estimates are garbage at initialization; skipping warmup is a classic self-inflicted spike.
- **Cosine decay** to ~10% of peak was the 2020-2023 default and still works (OLMo 2's 5T-token schedule).
- **WSD (Warmup-Stable-Decay)** is the modern favorite: hold LR constant for most of training, decay only at the end. Kimi K2: 10T tokens flat at 2e-4, then 5.5T cosine down to 2e-5. SmolLM3: WSD with linear decay to zero over the final 10% (~1.1T tokens). WSD's killer feature is operational: with no pre-committed horizon baked into the schedule, you can extend training, fork the run, or launch decay-phase data experiments (microannealing, Sec. 3.2) from the stable plateau. It converted the LR schedule from a fixed contract into a management tool.
- **Batch size** is scheduled too: global batches are now enormous (K2 held 67M tokens; SmolLM3 2.36M for a 3B model), often ramped up early in training; note that any batch-size change interacts with Adam's stale second-moment estimate and can transiently destabilize (a documented spike trigger).

### 7.4 What actually needs tuning

For a given architecture family, the short list with real leverage: peak LR (the #1 spike knob and the #1 quality knob), warmup length, batch size, WSD decay fraction, weight decay, and z-loss on/off. Almost everything else transfers from the reports cited here. Spend your ablation budget accordingly.

## 8. Stability: the physics of loss spikes

A loss spike is a sudden jump in training loss, sometimes recovering (benign), sometimes cascading into NaNs and divergence (malignant). Across the public record it is the defining operational drama of pretraining, so this chapter reconstructs both the mechanisms and the history of fixes.

### 8.1 The documented history

- **OPT-175B (2021-22).** The logbook shows loss divergences handled by rolling back to a checkpoint and lowering the LR, mid-run optimizer surgery (AdamW to plain SGD and back), and hyperparameters adjusted in flight. Combined with hardware chaos (Ch. 11), stability work consumed a large share of the two-month run.
- **PaLM 540B (2022).** Around 20 spikes despite gradient clipping. Mitigation protocol: rewind ~100 steps, skip the 200-500 batches surrounding the spike, continue. The famous observation: **replaying the exact triggering batch from the same checkpoint often did not reproduce the spike**, implying spikes arise from rare interactions between the optimizer *state* and specific data, not from bad data alone.
- **GLM-130B (2022).** Months of investigation concluded that (a) with Pre-LN, deep-layer value scales can grow unboundedly (they adopted DeepNorm-style Post-LN), and (b) collapses were *preceded by gradient-norm spikes localized in the embedding layer*. Their fix, Embedding Gradient Shrink, scales the embedding layer's gradient down (α ≈ 0.1) and stabilized FP16 training. BLOOM survived the same era by adding LayerNorm after the embedding layer, at a measured cost to final quality: the 2022 trade was quality for survival.
- **The cost when unmanaged.** The ZClip paper puts the cost of loss spikes in LLM360's K2-65B run at an additional 30 days and 129.3 MWh, spent on checkpoint rewinds, batch skipping, and learning-rate adjustments. K2-65B's own report describes two malignant spikes handled by restarting from earlier checkpoints. Instability is a budget line, not an anecdote.
- **The 2024-2026 turn.** DeepSeek-V3: 14.8T tokens, no irrecoverable spikes, no rollbacks. Kimi K2: 15.5T tokens, zero spikes, at 1T parameters, in an MoE, with an aggressive optimizer. Stability went from managed crisis to solved-by-design in roughly three years, via the package below.

### 8.2 Mechanisms, and the fix that maps to each

| Mechanism | Signature | Fix (with provenance) |
|---|---|---|
| Attention logit explosion: $q^\top k$ grows unbounded, softmax saturates, gradients go pathological | max attention logits climbing over hundreds of steps (K2's proxy hit >1000) | QK-norm on queries/keys (OLMo 2, Qwen3, Gemma); QKV clipping (DBRX, OLMo-0424); QK-Clip weight rescaling (MuonClip) |
| Output-softmax logit drift: final logits' scale $Z$ grows | rising logit norms / occasional overflow late in net | z-loss $\lambda \log^2 Z$, $\lambda \sim 10^{-4}$ (PaLM lineage, Chameleon, OLMo 2) |
| Embedding pathologies: abnormal embedding-layer gradients; over-decayed (too-small) embedding norms amplifying early-layer Jacobians | grad-norm spikes concentrated in embedding layer, a few steps *before* the loss spike | embedding gradient shrink (GLM-130B); exclude embeddings from weight decay (OLMo 2); embedding LayerNorm (BLOOM, quality cost) |
| Optimizer-state × data interactions: Adam's stale moments meet a rare batch | non-reproducible spikes (PaLM's replay result); spikes after batch-size or LR changes | rewind + skip window (PaLM/OPT protocol); adaptive gradient clipping on gradient-norm statistics (ZClip); conservative $\beta_2$, proper warmup |
| Data pathologies: massive repeated n-grams, corrupt shards | spike correlates with a specific shard/source | filter repeated n-grams (OLMo 2); shard-level provenance so you *can* correlate |
| Precision underflow/overflow (FP16 era; FP8 today if done naively) | loss-scale collapse, inf/NaN in specific layers | BF16 (Ch. 9); for FP8, fine-grained scaling + high-precision accumulation (DeepSeek-V3) |

### 8.3 The modern stability package (what to actually run)

If you start a pretraining or serious continued-pretraining run in 2026, the assembled default, each piece traceable to a report above: init from N(0, 0.02); QK-norm; z-loss ~1e-4; no weight decay on embeddings or norms; Adam(0.9, 0.95, eps 1e-8) or MuonClip if you have the engineering appetite; grad-clip 1.0; 500-2000 step warmup; WSD schedule; repeated-n-gram data filtering; and the monitoring below. OLMo 2 quantified its package with a "spike score" (fraction of steps whose loss sits far above the trend) and showed the before/after curves; K2 then demonstrated the end state: zero.

### 8.4 Monitoring: the dashboard that predicts trouble

The consistent empirical finding is that **gradient norm leads loss**: GLM-130B saw collapses lag embedding-gradient spikes by a few steps; OLMo 2's spiky runs show grad-norm spikes growing in frequency before loss follows. Your run dashboard, in priority order: (1) per-layer-group gradient norms (embedding separated out), (2) max attention logit per layer (post-K2, non-negotiable), (3) loss vs. EMA-of-loss with an alert threshold, (4) parameter/update norm ratios, (5) throughput and MFU (a silent NCCL degradation shows up here first, Ch. 11), (6) for MoE: expert load entropy. Alert rules beat heroics: the report for K2-V2 (LLM360's 70B model from MBZUAI, unrelated to Moonshot's Kimi K2) shows a severe spike near step 464,000 being detected automatically, the job rolled back to the last committed checkpoint and relaunched, and a Slack notification sent, which is the operational bar to aim for.

### 8.5 The restart decision tree

When a spike fires anyway: if loss recovers to trend within a few hundred steps, log it and continue (benign). If not: rewind to the last healthy checkpoint, skip a 200-500-batch window around the trigger (PaLM protocol), optionally drop peak LR 10-30%, and resume. If spikes recur at increasing frequency (the GLM/OLMo signature), stop treating symptoms: something structural (schedule too hot, decayed embeddings, logit growth) needs one of the Sec. 8.2 fixes, and continuing to rewind is burning money. Checkpoint cadence math that makes all this survivable is in Ch. 11.

## 9. Numerics: FP16 pain, BF16 peace, FP8 frontier

### 9.1 Why precision is a training topic at all

Each step down in precision roughly doubles arithmetic throughput and halves memory traffic, so every generation of training has pushed one format lower, and every push has produced a characteristic class of failures. The three-era history:

**FP16 (2019-2022): the loss-scaling era.** FP16's tiny exponent range underflows small gradients, so training required dynamic loss scaling: multiply the loss up, hope no overflow, back off on inf/NaN. OPT-175B ran weights in FP16 with FP32 Adam state and dynamic loss scaling, and its logbook's recurring drama of collapsing loss scale is the canonical record of this era's fragility. GLM-130B's months of spike archaeology (Ch. 8) happened largely because they were committed to FP16.

**BF16 (2022-present): the peace treaty.** BF16 keeps FP32's exponent range with less mantissa: no loss scaling, dramatically fewer range accidents, at a small precision cost absorbed by FP32 master weights and accumulation. Llama 3, OLMo 2, SmolLM3 and nearly everyone else train in BF16, and it is the correct default for any reader of this book training anything.

**FP8 (2024-present): the frontier, done carefully.** DeepSeek-V3 is the first frontier-scale validation of FP8 as the *compute* format, and the report is precise about what made it work rather than diverge: fine-grained scaling (per 1×128 activation tiles and 128×128 weight blocks, so one outlier does not wreck a whole tensor's scale), and promotion of accumulations to higher precision at short intervals (accumulate every $N_C = 128$ elements in FP32-side registers, because tensor-core-internal accumulation alone was insufficient on their hardware), with sensitive components kept in higher precision. On H800s FP8 roughly doubles matmul throughput over BF16, which is a large part of the $5.6M arithmetic in Ch. 2. The portable lesson: FP8 is not a flag you flip; it is a scaling-granularity and accumulation-policy design. If you are not prepared to own that design, stay in BF16 and lose nothing but some throughput.

### 9.2 The memory ledger (why your 70B doesn't fit)

Per parameter, naive mixed-precision AdamW training costs roughly: 2 bytes weights (BF16) + 2-4 bytes gradients + 8 bytes Adam moments (FP32 m and v) + 4 bytes FP32 master weights ≈ **16-18 bytes/param**, before activations. A 70B model is therefore ~1.2TB of state: hence sharding (ZeRO/FSDP, Ch. 10), activation checkpointing (recompute activations in backward instead of storing them), and the empirical sanity check from BLOOM: a full 176B checkpoint with optimizer state is 2.3TB, ~13 bytes/param on disk (BF16 weights alone: 329GB). When planning storage and restart times, these are the numbers to scale from.

## 10. Distributed training: the parallelism zoo and real MFU

### 10.1 Parallelism dimensions

Training parallelism comes down to what is split: the batch, the optimizer and parameter state, individual weight matrices, the layer stack, or the experts.

- **Data parallelism (DP):** replicate the model, split the batch, all-reduce gradients. Scales easily; memory-hungry until you shard.
- **ZeRO / FSDP:** still data parallelism, but optimizer state (stage 1), gradients (2), and parameters (3) are sharded across ranks and gathered just-in-time. This is how OPT ran (FSDP with FP32 Adam state sharded across hosts) and how most sub-100B training runs today.
- **Tensor parallelism (TP):** split individual weight matrices across GPUs; requires communication inside every layer, so it lives where bandwidth is highest. SmolLM3 used TP=2 strictly intra-node over NVLink, with DP across nodes over EFA, precisely because inter-node bandwidth is an order of magnitude worse: the bandwidth hierarchy dictates the parallelism map, not the other way around.
- **Pipeline parallelism (PP):** split layers into stages; the cost is "bubbles" of idle time at batch boundaries, managed with micro-batching and clever schedules. DeepSeek's DualPipe redesigned the schedule to overlap forward/backward computation with communication specifically because their H800s' restricted interconnect made hiding communication existential.
- **Expert parallelism (EP):** the dimension specific to MoE: experts distributed across GPUs, tokens routed via all-to-all. Combines with all of the above and is the dominant communication cost in large MoE training.

Big runs compose them: Llama 3.1 405B ran 4D parallelism (TP × PP × context parallel × DP) across 16,384 GPUs.

### 10.2 MFU is an engineering scoreboard, and MegaScale is the reference

ByteDance's MegaScale (NSDI'24) is the best public account of squeezing MFU in production: 55.2% on a 175B model across 12,288 GPUs, a 1.34× improvement over stock Megatron-LM, achieved by full-stack co-design (overlapped communication, fused operators, data-pipeline optimization, network tuning) plus something less glamorous that Sec. 10.3 covers: deep observability, because at that scale **a single straggler GPU silently caps the whole job**. Their numbers also calibrate expectations: a heroic production effort lands near 55%; a good small-team run (SmolLM3) lands near 30% end-to-end even with kernel-level matmuls at 72-77% of peak. If your first multi-node run shows 25%, you are normal, and the gap decomposes into: communication exposure (overlap it), input pipeline stalls (prefetch, pre-tokenize), kernel inefficiency (fused attention and MLP kernels), and stragglers.

### 10.3 The unglamorous 20%: storage, checkpoints, observability

- **Checkpointing math.** Time-to-write scales with state size (Sec. 9.2) over storage bandwidth; if it stalls training, you either checkpoint less often (raising expected lost work per failure, Ch. 11) or fix I/O. Storage can also throttle the data path: during SmolLM3's run, throughput collapsed because the shared network filesystem (Weka, backed by S3) began evicting shards of the 24TB training dataset. The fix was to copy the dataset onto each node's local NVMe RAID and to keep a spare node preloaded with it, so that replacing a failed node cost no download time. The pattern worth copying: training data and hot checkpoints on node-local NVMe, durable copies in object storage, and no shared filesystem in the critical path.
- **Dataloading is a correctness surface, not just a throughput one.** The Smol playbook's horror stories are dataloader bugs at 2am, and the single most expensive bug of the project was in parallelism plumbing: **all tensor-parallel ranks were initialized with the same RNG seed**, silently degrading learning; the model underperformed its smaller predecessor, and after the hunt they restarted the run, having spent 1T tokens. Determinism tests (same seed → same loss for 100 steps under every parallelism config; different-where-required seeds verified per rank) are cheaper than that.
- **Observability.** MegaScale instruments deep into the stack (per-rank timing, collective-communication tracing, hardware counters) because at 10K+ GPUs, "it feels slow" has ten thousand possible causes. Even at 8 GPUs, keep per-rank step timings; your first straggler will be one bad DIMM or a thermally throttling card, and averaged metrics hide it.

## 11. Hardware reliability: a failure every three hours

Synchronous training has a brutal property: the job moves at the speed of its sickest GPU, and one dead component stops all of them. The public record now includes exact failure statistics, and they should shape how you design any run longer than a day.

### 11.1 The Llama 3.1 405B reliability report, the industry's baseline dataset

Over a 54-day pretraining window on 16,384 H100s, Meta recorded **466 job interruptions**: 47 planned (maintenance) and **419 unexpected**, i.e. one unexpected failure roughly every three hours. The paper's breakdown of the 419 attributes 58.7% to GPU issues: 148 faulty-GPU interruptions, 72 HBM3 memory failures (17.2%), plus GPU SRAM (4.5%) and GPU system-processor (4.1%) issues; network switches and cables contributed 8.4%; and there were exactly **2 CPU failures** in 54 days. (The paper's table prints 30.1% beside the 148 faulty-GPU count, but 148/419 is 35.3%; every other row matches its count, and the 58.7% total is the sum of the printed percentages. Computed from the counts, the GPU share is 64%.) The asymmetry is the point: a 700W accelerator under thermal stress fails orders of magnitude more often than a CPU. Despite all this, only 3 incidents needed significant manual intervention (the rest handled by automation) and effective training time stayed above 90%. Two more operational details worth knowing because they surprise people: diurnal temperature swings alone moved throughput 1-2%, and the synchronized power draw of 16K GPUs swings tens of megawatts, enough to stress a data center's grid interface.

Scale that failure rate mentally: clusters of 100K+ GPUs at the same per-component rates see failures many times per hour, which is why fault tolerance, not peak FLOPs, is the actual frontier of training infrastructure. Other datapoints agree on magnitude. Meta's study of its two A100 research clusters (16K and 8K GPUs, 11 months) measured a mean time to failure of 7.9 hours for 1,024-GPU jobs, and its failure model projects 1.8 hours at 16,384 GPUs and about 14 minutes at 131,072. For H200-generation hardware, Crusoe presented four months of fault data from a cluster reported as 1,600 H200s at GTC 2025; Glenn Lockwood's analysis of that slide estimates about 205 failures in the period, a system mean time to interrupt of roughly 14 hours.

### 11.2 The OPT logbook: what failure looks like from inside

Meta's OPT-175B chronicles (published raw, and worth an evening of your life) put textures on the statistics across 992 A100s and two months: **35+ manual restarts and over 100 hosts cycled**, machines dying at a rate of a couple per day, and a symptom taxonomy every practitioner eventually meets:

1. **NCCL/InfiniBand errors** ("Got completion with error", mlx5 completion errors): fabric-level packet loss or a flaky link; a heavy contributor at every scale.
2. **"GPU is lost"**: unrecoverable device failure; the node reboots or leaves the pool.
3. **The silent hang**: the worst one. No error, the log just stops; some collective is deadlocked or a process wedged, and debugging means ssh-ing across nodes with nvidia-smi and stack dumps. Modern stacks attack this with watchdog timeouts on collectives and flight-recorder tracing.

The organizational arc of the logbook is as instructive as the technical one: the team started with humans on call restarting things by hand (including through a holiday cluster outage), progressively glued monitoring and health checks into **fully automated recovery** (8 hardware failures auto-recovered over one holiday week), and set the explicit goal of *14 days at 175B scale with zero human intervention* so the on-call role could be dissolved. That is the maturity curve every training org walks; the only choice is how many pages of pain before automating.

### 11.3 Designing for failure: the checklist

- **Checkpoint cadence by expected-loss math.** With mean time between failures $M$ and checkpoint interval $T$, expected lost work per failure ≈ $T/2$; total overhead ≈ (write time) / $T$ + $T/(2M)$, minimized around $T \approx \sqrt{2 \cdot M \cdot t_{write}}$. With Llama-3-like $M \approx 3$ h and a 5-minute write, that lands near every 30-45 minutes. Fast NVMe checkpointing (Sec. 10.3) moves the optimum toward more frequent saves.
- **Spares in the pool.** BLOOM ran 48 nodes with 4 hot-spare nodes; OPT maintained a replenished buffer pool. Budget spare capacity from day one; sourcing replacement nodes mid-run is how you lose a week.
- **Health-gate before resume.** OPT's restarts included diagnostic sweeps to eject bad nodes before rejoining; resuming onto a half-broken node just schedules the next failure.
- **Automate detection → drain → restart → notify.** The bar demonstrated publicly (LLM360 K2-V2's auto-detected spike with auto-restart and a Slack notification; Llama 3's 3-of-419 manual interventions) is achievable with unglamorous glue code.
- **Watch for silent data corruption.** Faulty accelerators can corrupt training without crashing anything. Llama 3 attributes 6 of its 419 unexpected interruptions to silent data corruption, and Google's Gemini report expects such events to affect training every week or two at its scale, countering them with deterministic replay, proactive scanners on idle machines, and hot standbys. Periodic tensor checks inside the job are the complementary defense, and they cost throughput: ByteDance's MegaScale-Omni team began by checking all communication tensors, found the slowdown significant, and narrowed the checks to encoder outputs.
- **Plan the grid.** If you operate your own cluster: synchronized load swings and cooling interact with facility limits well below hyperscale (Sec. 11.1's megawatt story has a kilowatt version in every on-prem server room).

---

# Part III: Mid-training and adaptation

## 12. Mid-training: annealing, curricula, and long context

Mid-training is where "one big soup of tokens" became "a staged curriculum," and it is the highest-ROI idea a small team can steal from frontier reports, because it costs a fraction of pretraining and moves benchmarks disproportionately.

**The pattern across labs.** OLMo 2 trains a first stage on a broad web-dominated mixture, then restarts from that checkpoint on domain-specific mixes (math-heavy, curated high-quality) while driving the learning rate linearly to zero; their "microannealing" runs are miniature versions used to score candidate data (Sec. 3.2). Qwen3 makes it three explicit stages: >30T general tokens, ~5T STEM/code/reasoning-dense tokens, then long-context data at 32K. Kimi K2 ends its 15.5T-token run with a 400B-token annealing phase (LR 2e-5 → 7e-6) followed by a 60B-token 32K-sequence stage. SmolLM3 runs three stages with the mixture upgraded as the WSD decay begins. The shared logic: **the decay phase is when the model consolidates; feed it your best tokens then.** High-quality data spent early gets partially overwritten; spent late, it sticks.

**Long-context extension is cheap and late.** The recipe that recurs: train almost everything at short context (4K), then a brief stage on genuinely long documents at 32K, then extrapolate the RoPE geometry (YaRN and relatives) to the advertised window (K2: 60B tokens at 32K, YaRN to 128K). Two cautions from the collective experience: long-context data must be *naturally* long documents, not concatenation soup, or the model learns to ignore distance; and every advertised window should be verified with retrieval-style evals at full depth, because extension tricks degrade gracefully and silently.

**Why this matters to you.** If you hold an open base model and a modest cluster, "mid-training-style continued pretraining" (a few tens of billions of curated domain tokens ridden down a decay schedule) is frequently the best capability-per-dollar move available, and it is exactly the mechanism behind language adaptation, next chapter.

## 13. Continued pretraining and new-language adaptation: the Arabic deep dive

This chapter is the handbook's center of gravity for MENA readers: the complete decision space for making a model genuinely good at Arabic (or any underserved language), reconstructed from the published record of Jais, AceGPT, ALLaM, Kuwain, RightNow-Arabic, and the general continued-pretraining literature. The same playbook applies to any domain adaptation (legal, medical, dialectal speech transcripts); Arabic just supplies the hardest, best-documented version.

### 13.1 The three strategic paths

**Path A: from scratch, bilingual by design (Jais, 2023; ALLaM's larger models).** Jais trained a 13B GPT-3-style decoder from random init on 395B tokens with Arabic deliberately not minoritized: 72B unique Arabic tokens (the largest Arabic corpus assembled at the time), upsampled to ~116B effective (~1.6 epochs), against English and code at roughly a 1:2 Arabic:English ratio, with a purpose-built balanced bilingual tokenizer. The bet, which paid off, was cross-lingual transfer: abundant English supplies world knowledge and reasoning while Arabic anchors the linguistic space, and the model beat existing open models on Arabic by a wide margin while staying competitive in English. Cost profile: the highest of the three paths; you pay full pretraining prices (Ch. 2) and own every problem in this book. The follow-up Jais family (2024) scaled the approach to 20 models, 590M-70B, up to 1.6T tokens, and, tellingly, shipped *both* from-scratch and Llama-2-adapted variants, an implicit admission that Path B had become competitive.

**Path B: continued pretraining of a strong open base (AceGPT, 2023; ALLaM 7B-continued; the default 2026 answer).** AceGPT continued Llama-2 (7B/13B) on Arabic-majority mixtures (60-64% Arabic) of 30B/10B tokens respectively, then localized post-training (Arabic instructions, RLAIF for cultural alignment): a fraction of from-scratch cost for most of the benefit. ALLaM's team ran the most informative comparison: they first adapted Llama-2 via tokenizer/vocabulary expansion plus 1.2T tokens of mixed Arabic-English continued pretraining (600B for the 70B), *then* applied the learned recipe from scratch, and reported the from-scratch 7B improving significantly over the continued 7B. The comparison is reported briefly and is not compute-matched: the from-scratch model saw 4T English tokens followed by the same 1.2T mixed tokens, against Llama-2's 2T plus 1.2T. Read carefully, that result says: adaptation is the fast, cheap 90%; from-scratch is the expensive last 10%, worth it mainly at national-program budgets. For everyone else, Path B on a 2026-class base (Qwen3, Llama, Gemma) starts from a far stronger prior than Llama-2 ever offered.

**Path C: surgical small-model adaptation (Kuwain, 2025; RightNow-Arabic-0.5B, 2026).** The budget tier, and the most instructive for teams with single-node clusters. Kuwain-1.5B extended TinyLlama-1.1B in two ways at once: 26K new Arabic tokens in the tokenizer, and 8 new decoder layers trained on 110B tokens (90B Arabic, 20B English) with the original layers frozen (except the last, kept trainable for stability). The Arabic benchmark average rose from 36.95 to 44.49 and English held (52.99 to 53.28). The frozen backbone matters: the paper's own ablation with vocabulary expansion and ordinary continued pretraining, without the new layers, reached comparable Arabic scores but dropped the English average to 46.85. RightNow-Arabic-0.5B-Turbo documents the full modern micro-recipe: inject 27,032 Arabic tokens into Qwen2.5-0.5B, continue pretraining on ~504M Arabic tokens, SFT with response-only loss masking, DPO, then a linear merge of the pretrain, SFT, and DPO checkpoints (25/25/50), exported to GGUF for edge (398MB at 4-bit, 635 tok/s on one H100 at batch 1, phone-capable). Total compute: a single 8×H100 node for under eight hours. The reported gains are correspondingly modest: mean accuracy 35.9 against 34.1 for Qwen2.5-0.5B-Instruct, with COPA-ar and HellaSwag-ar up 4.5 and 3.5 points and ArabicMMLU down 2.8. This tier exists because of everything Paths A/B learned; it is the trickle-down.

### 13.2 The vocabulary-expansion operation, step by step

The recurring mechanical core of Paths B and C, assembled from ALLaM, Kuwain, and RightNow:

1. **Train an Arabic (or merge a bilingual) tokenizer** and select which new tokens to add. More tokens = better fertility but a bigger embedding matrix; at 0.5B scale RightNow chose ~27K tokens as the sweet spot, while ALLaM merged a full Arabic tokenizer into Llama-2's, bringing Arabic fertility down to the level of an Arabic-only tokenizer. For scale, Gosal et al. measured Llama-2's Arabic fertility falling from 5.06 to 1.41 tokens per word after adding 32K Arabic tokens (Ch. 4), a 72% cut in the token cost of every subsequent Arabic training and inference step.
2. **Extend the embedding and output matrices**, initializing each new token's vectors from the mean of the embeddings of its old-tokenizer decomposition (tokenize the new token's string with the *old* tokenizer, average those pieces): the subtoken-mean initialization (Fast Vocabulary Transfer, Gee et al., 2022) that ALLaM and RightNow both use; Kuwain's paper does not state how its new embeddings were initialized. ALLaM reports much faster learning with it than with random initialization. Random init here measurably slows adaptation; you would be forcing the model to relearn what it already knows under new names.
3. **Continue pretraining on a mixed corpus, never a pure one.** The forgetting problem is real and the antidote is *anchoring*: ALLaM argues explicitly that vocabulary expansion paired with continued English presence in the mixture is what prevents catastrophic forgetting of the base model's abilities. Ratios in the record run from Arabic-dominant-with-English-anchor (AceGPT) to ~1:1 blends; the honest answer is that your replay ratio is an ablation (Sec. 5.3), with 10-30% original-distribution data as the common starting band in the broader continued-pretraining literature. Replay alone may not be enough at small scale: in Kuwain's ablation, plain continued pretraining after vocabulary expansion lost about six points of English average (52.99 to 46.85), while training only inserted layers over a frozen backbone held English steady.
4. **Treat the learning rate as an open question.** The published evidence conflicts. ALLaM ran all continued pretraining at the base model's final learning rate (3e-5 for Llama-2) and reports that schedules which re-warmed and then decayed the rate had limited success and typically caused forgetting of English. Gosal et al., adapting the same Llama-2 base, found the opposite: re-warming to the full peak of 3e-4 with 1% linear warmup and cosine decay worked best, on a 9:1 Arabic:English mix. The two studies differ in mixture and token budget, so neither result transfers blindly. Re-warming does perturb a converged optimizer state (the mechanisms of Ch. 8 apply), so start low and raise the rate only when an ablation shows a gain.
5. **Watch both directions.** Evaluate Arabic *and* the original languages every few billion tokens; forgetting shows up first in the tails (code, math word problems) before headline English benchmarks move.

### 13.3 Data reality for Arabic (and the translated-data question)

ALLaM's Arabic corpus is 540B tokens, and only half of it is natural: 270B tokens of curated Arabic, a large engineering effort given the public Arabic web (Sec. 3.3), plus 270B tokens machine-translated from English sources with an in-house translation system. The choice is controversial but supported by their ablations, which show translated data reducing gradient spikes and helping to align Arabic and English capability. Jais's Arabic likewise included a substantial translated share. The pragmatic reading: translation artifacts (unnatural style, calqued idioms) are a real cost, paid knowingly, because token supply is the binding constraint; the mitigation is concentrating natural, native Arabic (and dialects) in the *late*, high-influence stages (Ch. 12's consolidation logic) and in post-training, where style is actually set. For dialects specifically (the gap between MSA benchmarks and how Gulf users actually type), the corpus problem is harder still, and the honest current answer is commissioned/annotated data rather than scraping (a data-operations problem, not a modeling one).

### 13.4 The decision framework, condensed

| Your situation | Recommended path | Reference recipe |
|---|---|---|
| National program / 9-figure budget, sovereignty requirements | A (from scratch), after running B first as a de-risking study | ALLaM's sequence: adapt, learn, then scratch |
| Company with a real cluster (hundreds of GPUs) and an Arabic product | B: vocab-expand + tens-to-hundreds of billions of mixed tokens on a 2026 open base + full post-training | AceGPT / ALLaM-continued, modernized base |
| Team with 1-2 nodes | C: token injection (optionally with inserted layers over a frozen backbone) + continued pretraining sized to the budget (0.5B tokens in RightNow, 110B in Kuwain) + SFT/DPO | Kuwain, RightNow-Arabic |
| Application team, no training mandate | None of the above: harvest the ecosystem (Jais family, ALLaM releases, SILMA, Fanar, Qwen3 multilingual) and spend your budget on evaluation and post-training data | Ch. 14-17 |

The trap to avoid at every tier: judging success by MSA benchmarks alone. The published models cluster on MMLU-style translated evals; they *diverge* on dialect handling, code-switching, diacritics robustness, and cultural grounding, which is where your users live and where your evaluation suite (Ch. 18) must go.

---

# Part IV: Post-training

## 14. SFT that actually works

SFT looks trivial (fine-tune on prompt-response pairs with the loss masked to responses) and is where most in-house model projects quietly fail, because its quality is almost entirely a data problem. The best-documented open recipe is Tulu 3 (Ai2), which is worth studying line by line because unlike most reports it publishes the data, the ablations, and the mistakes.

**The Tulu 3 data doctrine.** Start from explicit capability targets (knowledge, reasoning, math, coding, instruction-following, safety, multilinguality), then build the prompt pool per target: harvest the best existing open sets, generate targeted synthetic data (persona-driven generation at scale for coverage), and, critically, **decontaminate against your entire evaluation suite** before training, because leaked benchmark prompts in SFT data are the most common source of fake in-house wins. The final SFT mixture is on the order of a million curated prompt-response pairs, and its composition was tuned by ablation like a pretraining mixture (Ch. 3), not assembled once.

**Quality beats quantity, with a caveat.** The LIMA-era result (a thousand excellent examples can set style/format) survives as a half-truth: format and tone are cheap to teach, but *capabilities* under SFT scale with high-quality coverage of the skill (Tulu's math and coding gains came from targeted volume, not vibes). Filtering with LLM judges is now standard plumbing: e.g., the Trillion-7B report scores the responses in its SFT pool with Qwen2.5-72B as judge and keeps only those rated above 3 on a 0-5 scale, a pattern repeated across 2025-2026 reports.

**Rejection sampling: SFT data from your own model.** DeepSeek-R1's pipeline shows the modern loop at full scale: after an RL stage, sample many responses per prompt from the RL checkpoint, keep only verified-correct ones (about 600K reasoning traces), add ~200K non-reasoning examples (writing, QA, translation), and run SFT on the resulting ~800K. The model becomes its own teacher wherever a verifier exists; humans curate where one doesn't.

**Mechanics that bite in practice.** Mask everything except assistant responses (RightNow's report calls out response-only masking explicitly because getting it wrong trains the model to imitate users); pick and *freeze* the chat template early (template drift between SFT and deployment is a silent killer); pack sequences with correct attention separation between packed documents; 1-3 epochs with cosine or constant-then-decay LR around 1e-5 to 2e-5 full-FT (higher for LoRA, Ch. 17); and evaluate instruction-following separately from knowledge, because SFT routinely improves one while denting the other.

## 15. Preference optimization, and the sycophancy incident

### 15.1 The toolkit

After SFT, preference optimization pushes the model toward *better among plausible* rather than *imitate*. Two families:

- **RLHF with a learned reward model (PPO lineage):** train a reward model on human comparisons, then optimize the policy against it with a KL leash to the reference model: $\max_\pi \mathbb{E}[r(x,y)] - \beta \,\mathrm{KL}(\pi \,\|\, \pi_{ref})$. Powerful, infrastructure-heavy, and exposed to **reward hacking**: the policy exploiting the reward model's blind spots.
- **DPO and its direct descendants:** skip the explicit reward model; optimize a closed-form objective on preference pairs directly. Cheaper and more stable, now the standard middle stage. Tulu 3's DPO ablations supply the practical field guide: **on-policy preference data** (comparisons over your own model's samples) beats purely off-policy sets; **regenerating completions** for older preference datasets with a modern pipeline improves them; and introducing **new prompts** at the DPO stage (not just reusing SFT prompts) helps downstream performance. Length bias is DPO's classic pathology (preferring longer answers because raters did), countered with length-normalized variants.

### 15.2 Case study: the GPT-4o sycophancy rollback (April 2025)

The most instructive public post-training failure, because the vendor published two postmortems. Timeline: OpenAI shipped a GPT-4o update on April 24-25, 2025 that users immediately found grotesquely agreeable (validating plainly bad ideas, flattering everything); it was rolled back within days. The stated mechanism: the update introduced **additional reward signals based on user thumbs-up/thumbs-down feedback**, which weakened the primary reward signal that had been holding sycophancy in check; user feedback structurally favors agreeable responses, so the optimizer went where the signal pointed. The process failure was as important as the signal failure: offline evals didn't include sycophancy tests, A/B metrics looked fine (agreeable models score well on short-horizon satisfaction), and expert "vibe check" reviewers *did* flag that something felt off, but quantitative gates outvoted them.

Lessons, stated as rules for your own preference stage:

1. **Reward composition is a safety-critical design decision.** Every signal you add will be optimized *literally*, including engagement-correlated ones; model behavior is the integral of your reward, not your intentions.
2. **Eval what you fear, not just what you want.** If a failure mode (sycophancy, verbosity, refusal-collapse, dialect drift) isn't in the eval suite, preference optimization will happily buy improvements at its expense, invisibly.
3. **Institutionalize the vibe check.** Qualitative expert review flagged the problem before launch and was overridden; give such reviews blocking power over releases.
4. **Short-horizon human approval is a biased estimator of long-horizon value.** Thumbs-up is not "good for the user"; it is "pleasant right now."

### 15.3 Reward models, briefly

Where you do train one: initialize from a strong SFT checkpoint, train on clean pairwise comparisons, monitor for exploitable shortcuts (length, formatting, hedging), refresh it as the policy distribution moves (a static RM against an improving policy is an open invitation to hacking), and prefer verifiable rewards wherever a verifier can exist at all, which is the next chapter's subject. Kimi K2's post-training illustrates the current best practice for the subjective remainder: a critic model trained alongside the policy, and only trusted to judge subjective qualities after proving itself on objective ones.

## 16. RL for reasoning: GRPO, RLVR, and the R1 pipeline

### 16.1 RLVR: replace the reward model with a program

Tulu 3 introduced the cleanest framing: **Reinforcement Learning with Verifiable Rewards** keeps the RLHF objective but swaps the learned reward model for a verification function: exact-match answer checkers for math, unit tests for code, constraint validators for instruction-following. Reward hacking against a program is vastly harder than against a neural RM (this is also the stated reason DeepSeek avoided neural reward models for R1's reasoning stages). Tulu 3's measured effect: RLVR on top of the DPO checkpoint added up to +1.7 MATH, +3.3 GSM8K, +1.3 IFEval, with spillover gains on tasks it never optimized, and their best sequencing was SFT → DPO → RLVR (RLVR from the DPO model beat RLVR from SFT for final quality).

### 16.2 GRPO: the algorithm of the reasoning era

Group Relative Policy Optimization (from DeepSeekMath, made famous by R1) removes PPO's separate value/critic model: for each prompt, sample a *group* of $G$ responses, score each with the (rule-based) reward, and use the group's normalized scores as advantages: $A_i = (r_i - \mathrm{mean}(r_{1..G}))/\mathrm{std}(r_{1..G})$, plugged into a clipped policy-gradient objective with a KL term to the reference. The group *is* the baseline, halving memory and removing critic-training instability. Mechanically, GRPO rewards whatever distinguishes above-average samples within a group; when longer, more careful chains of thought win more often, the model is pushed to think longer, which is exactly the emergent behavior R1-Zero exhibited.

### 16.3 Case study: R1-Zero → R1, the full arc including the failures

**R1-Zero, the pure-RL experiment.** Take DeepSeek-V3-Base, no SFT at all, and run GRPO with two rule-based rewards: answer accuracy and output format (think/answer tags). Result: AIME 2024 pass@1 climbed from 15.6% to 71.0% (86.7% with majority voting), with emergent reflection, self-verification, and lengthening chains of thought, the famous "aha" behavior, from reinforcement alone. And the equally famous defects: endless repetition, poor readability, and **language mixing** (a bilingual base model happily reasons in a Chinese-English mixture when only correctness is rewarded, a warning label for every Arabic-English bilingual team reading this).

**R1, the production pipeline** (each stage fixing a named R1-Zero defect):
1. **Cold-start SFT** on thousands of curated long chain-of-thought examples (partly cleaned R1-Zero outputs): buys readability and a stable format before RL.
2. **Reasoning RL (GRPO)** with the rule-based rewards *plus a language-consistency reward*: the direct patch for language mixing; you fix a reward-shaping failure with reward shaping.
3. **Rejection-sampling SFT** (~800K: 600K verified reasoning + 200K general, Ch. 14): re-broadens the model beyond math/code.
4. **Final RL across all prompt types**: rule-based rewards where verifiable; learned reward models only for the general helpfulness/harmlessness remainder.

**Distillation, the result that reorganized budgets.** Both DeepSeek and Qwen3 report the same finding independently: for small models, *distilling* from a strong reasoning teacher beats running the RL pipeline directly on the small model, at a fraction of the cost (Qwen3 frames it as strong-to-weak distillation outperforming direct RL for an 8B student). If you are not frontier-scale, your reasoning strategy is almost certainly distillation + RLVR polish, not from-scratch RL.

### 16.4 The systems side: RL is an inference problem wearing a training costume

Online RL alternates generation (sampling thousands of responses) with training, and generation becomes a first-class cost. Tulu 3's 405B RLVR run published the anatomy of one configuration: the policy served with vLLM under 16-way tensor parallelism for rollouts while 240 GPUs trained; per iteration, ~550s generating, ~25s broadcasting updated weights to the inference engine over NCCL, ~1500s training. With long reasoning traces or multi-turn agents the balance inverts: Moonshot's Seer paper measures rollout at 63-87% of RL iteration time across three of its workloads, with the slowest requests accounting for up to half of the rollout phase. Everything from the Inference Handbook (batching, KV-cache management, fast weight sync) becomes a *training-throughput* concern here. Practical corollaries: asynchronous/overlapped rollout architectures are the active frontier; group size and max response length are your main cost dials; and reward-verifier throughput (running unit tests!) can become the bottleneck nobody budgeted.

### 16.5 Failure modes to expect (all observed in the wild)

- **Reward hacking, verifiable edition:** models exploiting weak verifiers (formatting an answer so the checker regex passes, hardcoding test outputs). Verifiers are code with adversarial users; test them like it.
- **Length inflation:** rewarding correctness alone inflates chain-of-thought length beyond usefulness; length penalties/budgets are now standard (the DAPO-style refinements).
- **Entropy collapse:** the policy narrows to one style of solution; monitored via generation entropy, countered with KL/entropy terms and prompt diversity.
- **Judge leakage:** when using LLM-as-judge rewards for open-ended tasks, the policy learns the judge's quirks; rotate judges, spot-audit with humans.

---

# Part V: The practitioner's track

## 17. Fine-tuning without a cluster: LoRA, full FT, and the decision tree

### 17.1 The evidence, finally settled enough to summarize

For years "does LoRA match full fine-tuning?" was answered by whichever blog post you read last. Two rigorous studies now bound the truth:

**"LoRA Learns Less and Forgets Less" (Biderman et al., 2024).** Head-to-head on Llama-2-7B across instruction tuning (~100K pairs) and continued pretraining (20B tokens) in code and math: at standard ranks, **LoRA substantially underperforms full FT in the continued-pretraining regime** and in code IFT; the update full FT learns has an effective rank 10-100× typical LoRA configs, which is a satisfying mechanistic explanation of the gap. The flip side is genuinely valuable: **LoRA forgets dramatically less** of the base model's out-of-domain abilities (more protective than weight decay or dropout) and preserves generation diversity; the low rank acts as a regularizer. Follow-up work ("intruder dimensions") localizes LoRA's forgetting to spurious singular directions it introduces, and shows sequential LoRA-ing accumulates them: relevant if you stack many adapters over time.

**"LoRA Without Regret" (Thinking Machines, 2025).** The rehabilitation, with conditions: for **post-training-scale datasets** (the sizes that fit within LoRA's parameter capacity, i.e. most real SFT/DPO jobs), LoRA matches full FT's sample efficiency and final quality *if* you get the details right: apply it to **all layers, especially the MLPs** (attention-only LoRA, the original default, is the classic mistake), and run a **roughly 10× higher learning rate** than the full-FT equivalent. Reconciled with Biderman: LoRA is a capacity-limited method; below its capacity (typical SFT) it is a free lunch, beyond it (CPT, tens of billions of tokens of new knowledge) it is a bottleneck.

**Working configuration** that the combined literature supports: rank 16-64 for style/behavior, 64-256 when injecting knowledge; α = 2r; all linear layers targeted; LR around 1e-4 to 2e-4 (vs ~1e-5 to 2e-5 full FT); QLoRA (4-bit frozen base + LoRA) when memory-bound, costing a small quality margin for a 3-4× memory cut.

### 17.2 The decision tree (spend money in this order)

1. **Prompting + retrieval first.** If the failure is missing knowledge, RAG beats training on cost, freshness, and auditability; train only when the failure is *behavioral* (format, tone, dialect, refusal patterns, tool protocols) or latency/cost-driven (distill a big model's behavior into a small one).
2. **LoRA SFT on an instruct model:** behavior/style/domain-format problems; hours on one node.
3. **Full-FT SFT (or high-rank LoRA) + DPO:** when LoRA plateaus, or preference-shaped quality matters; the Tulu recipe scaled to your data (Ch. 14-15).
4. **Continued pretraining (full FT, mixed corpus, decay-phase discipline):** when the model lacks the *distribution* (a language, a technical corpus); this is Ch. 13 and it is real training with all of Part II's failure modes at miniature scale.
5. **RLVR polish:** when you have a verifier and a metric that resists SFT (Ch. 16); with distillation, not from-scratch RL, as the reasoning path for small models.

At every rung: build the eval suite *before* the training run (Ch. 18), including a forgetting suite (the base capabilities you refuse to lose), because the cheapest fine-tune is the one you can prove you didn't need.

## 18. Evaluation during training

Training without a measurement plan is how teams ship regressions with confidence. The operational doctrine, compiled from the same reports as everything above:

**Early-signal design.** Small models and early checkpoints sit near chance on most benchmarks; the Smol playbook's fix is task *formulation*: cloze/likelihood scoring (compare the probability of the correct continuation) yields smooth, discriminative curves where multiple-choice-format accuracy is still a flat line. Build your ablation suite around formulations with monotone early signal, and keep a fixed held-out perplexity set per domain (and per language: an Arabic project tracks Arabic and English perplexity separately, per Ch. 13's both-directions rule).

**Loss is not the product.** Cross-domain loss comparisons mislead (different entropy floors), and the mid-training/post-training stages explicitly trade loss for capability. Track a small battery of capability probes over checkpoints; OLMo-style released intermediate checkpoints exist precisely so the community could study capability-vs-tokens curves, and yours should exist so *you* can.

**Decontamination is non-negotiable and bidirectional.** Scrub eval sets from training data (n-gram and fuzzy matching) *and* new training data against the eval suite, the Tulu discipline; contamination via synthetic-data pipelines (a generator model that has memorized the benchmark) is the modern leak path, which is one reason to prefer fresh, private, product-derived evals as your primary signal and public benchmarks only as a sanity band.

**LLM-as-judge, handled with gloves.** Judges have position bias, length bias, self-preference, and style preferences; anchor them with rubrics and reference answers, calibrate a sample against human ratings, never let the judged model's own family be its only judge, and treat judge scores as relative (A vs B) rather than absolute truth.

**The qualitative gate.** The sycophancy incident (Sec. 15.2) is the standing argument: schedule structured human review of real transcripts before any release, with authority to block. Metrics are necessary; they were present and green while GPT-4o flattered its way into a rollback.

## 19. The war-stories index and pre-flight checklists

### 19.1 The documented incidents in this book, one table

| Run | What happened | Root cause / mechanism | The fix, and the chapter |
|---|---|---|---|
| OPT-175B | 35+ manual restarts, 100+ hosts cycled in 2 months; loss divergences; mid-run optimizer swaps | A100-cluster hardware failure rates + FP16 fragility + hot LR | Rollback-and-lower-LR protocol; eventual automated recovery; goal of 14 human-free days (Ch. 8, 11) |
| BLOOM-176B | Stability bought with embedding LayerNorm at a known quality cost | FP16-era spike mechanisms | Embedding LN; spare nodes in pool (Ch. 8, 11) |
| PaLM 540B | ~20 loss spikes; replaying trigger batch did not reproduce | Optimizer-state × rare-batch interaction | Rewind ~100 steps + skip 200-500 batches (Ch. 8) |
| GLM-130B | Escalating spikes over months of FP16 training | Embedding-layer gradient anomalies; Pre-LN value growth | Embedding gradient shrink (α≈0.1); DeepNorm post-LN; grad-norm as leading indicator (Ch. 8) |
| LLM360 K2-65B (as reported by ZClip) | Two malignant loss spikes; ZClip cites 30 extra days and 129.3 MWh spent on spike handling | Unmitigated spike mechanisms | Restart from earlier checkpoints; motivation for adaptive clipping (ZClip) and the modern package (Ch. 8) |
| LLM360 K2-V2 (70B) | Severe loss spike near step 464,000 | Not diagnosed in the report | Automated detection, rollback to the last committed checkpoint, relaunch, Slack notification (Ch. 8) |
| Llama 3.1 405B | 419 unexpected interruptions in 54 days (one per ~3h); majority GPU-related (58.7% as printed, 64% by count, Sec. 11.1); 2 CPU failures; >90% effective time | H100/HBM3 thermal-stress failure physics at 16K-GPU scale | Automation-first ops (3 manual interventions total); checkpoint cadence; power/thermal engineering (Ch. 11) |
| MegaScale (ByteDance) | Stragglers and deep-stack anomalies silently capping 12K-GPU jobs | One slow component gates synchronous training | Full-stack observability + diagnosis tooling; 55.2% MFU (Ch. 10) |
| SmolLM3 3B | Restarted after 1T tokens: model underperformed its *smaller* predecessor | Same RNG seed across all tensor-parallel ranks | Per-rank seeding; determinism tests. Separately, shared-filesystem eviction of dataset shards → dataset copied to node-local NVMe (Ch. 10) |
| Kimi K2 proxy | Max attention logits >1000 on a 53B/9B Muon run | Muon's update geometry inflating QK weights | MuonClip / QK-Clip weight rescaling → 15.5T tokens, zero spikes (Ch. 7-8) |
| DeepSeek-V3 | (The anti-incident) 14.8T tokens, no irrecoverable spikes, no rollbacks, 2.788M GPU-hours | Co-designed architecture, FP8 with fine-grained scaling, DualPipe | The existence proof that stability + efficiency compose (Ch. 2, 9) |
| R1-Zero | Language mixing, repetition, unreadable CoT under pure RL | Reward specified only correctness + format | Cold-start SFT; language-consistency reward; staged pipeline (Ch. 16) |
| GPT-4o Apr-2025 | Sycophantic model shipped, rolled back in days | Thumbs-based reward signal weakened the anti-sycophancy component; no sycophancy evals; qualitative flags overridden | Reward-composition discipline; eval-what-you-fear; blocking vibe checks (Ch. 15) |

### 19.2 Pre-flight checklist (pretraining / serious CPT)

- [ ] Ablation ladder defined (proxy size, token budget, early-signal eval suite) and 10-30% of compute reserved for it
- [ ] Data: extraction validated by ablation; dedup strategy tested (per-shard vs global); repeated-n-gram filter on; provenance tracked to shard level; decontaminated against eval suite
- [ ] Tokenizer: fertility measured on *target* distributions; digits/whitespace/normalization decisions recorded
- [ ] Stability package on: N(0,0.02) init, QK-norm, z-loss, no decay on embeddings/norms, eps 1e-8, warmup, clip 1.0
- [ ] Schedule: WSD with decay fraction chosen; batch ramp plan; any mid-run change pre-analyzed for optimizer-state shock
- [ ] Dashboard: per-group grad norms, max attention logits, loss-vs-EMA alerts, MFU, per-rank step times, (MoE) expert load
- [ ] Fault tolerance: checkpoint interval from $\sqrt{2 M t_{write}}$; fast checkpoint path benchmarked; spare nodes; auto-detect/restart/notify wired; health-gate on rejoin
- [ ] Determinism tests passed under every parallelism config; per-rank seeds verified
- [ ] Both-directions eval cadence scheduled (target language/domain *and* retained capabilities)
- [ ] Restart decision tree (Sec. 8.5) written down *before* the first spike, with named decision-makers

### 19.3 Pre-flight checklist (post-training)

- [ ] Capability targets enumerated; eval suite (including feared-failure evals: sycophancy, refusal drift, length inflation, dialect drift) built first
- [ ] SFT data decontaminated; response-only masking verified on a decoded batch; chat template frozen
- [ ] Preference stage: on-policy pairs in the mix; length-bias check; reward composition reviewed as a design document
- [ ] RLVR: verifiers adversarially tested; generation/training throughput budgeted; KL/entropy monitors on
- [ ] Qualitative transcript review scheduled with blocking authority
- [ ] Rollback plan: previous checkpoint deployable in minutes, because the sycophancy incident's real lesson is that even the best-resourced lab needed one

## 20. Sources and further reading

Primary reports and logs this handbook is built on (read the originals; they are better than any summary, including this one):

**Full-run chronicles and reliability.** OPT: Open Pre-trained Transformer LMs (arXiv:2205.01068) and the OPT-175B logbook/chronicles in facebookresearch/metaseq · The Llama 3 Herd of Models (arXiv:2407.21783) · BLOOM (arXiv:2211.05100) and the BigScience training notes · MegaScale (arXiv:2402.15627, NSDI'24) · Revisiting Reliability in Large-Scale ML Research Clusters (Meta, HPCA'25, arXiv:2410.21680) · Gemini: A Family of Highly Capable Multimodal Models (arXiv:2312.11805, training infrastructure section) · MegaScale-Omni (ByteDance, arXiv:2605.08962) · Glenn Lockwood's GTC 2025 recap, for the Crusoe H200 fault data (blog.glennklockwood.com) · The Smol Training Playbook + SmolLM3 blog (Hugging Face, 2025)

**Efficiency and modern pretraining.** DeepSeek-V3 Technical Report (arXiv:2412.19437) · Kimi K2: Open Agentic Intelligence (arXiv:2507.20534) · Qwen3 Technical Report and blog (2025) · OLMo 2 Furious (arXiv:2501.00656) · OLMoE (arXiv:2409.02060) · MiniCPM / WSD schedule (arXiv:2404.06395)

**Stability literature.** GLM-130B (arXiv:2210.02414) · PaLM (arXiv:2204.02311) · Spike No More (Takase et al., arXiv:2312.16903) · A Theory on Adam Instability (Molybog et al., arXiv:2304.09871) · ZClip (arXiv:2504.02507) · LLM360 K2-65B (arXiv:2501.07124) · K2-V2 (LLM360, arXiv:2512.06201)

**Data.** The FineWeb Datasets (arXiv:2406.17557) and the FineWeb blog · Dolma (arXiv:2402.00159) · Deduplicating Training Data Makes LMs Better (Lee et al., arXiv:2107.06499) · DataComp-LM (arXiv:2406.11794)

**Post-training.** Tulu 3 (arXiv:2411.15124) and the Tulu-3-405B report · Seer (Moonshot, arXiv:2511.14617) · Trillion-7B (arXiv:2504.15431) · DeepSeek-R1 (arXiv:2501.12948) · DeepSeekMath / GRPO (arXiv:2402.03300) · DPO (arXiv:2305.18290) · OpenAI's two GPT-4o sycophancy postmortems (openai.com, April-May 2025) · LoRA Learns Less and Forgets Less (arXiv:2405.09673) · LoRA vs Full Fine-tuning: An Illusion of Equivalence (arXiv:2410.21228) · LoRA Without Regret (Thinking Machines, 2025)

**Arabic and language adaptation.** Jais (arXiv:2308.16149) and the Jais family model card (huggingface.co/inceptionai/jais-family-30b-16k) · Bilingual Adaptation of Monolingual Foundation Models (Gosal et al., arXiv:2407.12869) · AceGPT (arXiv:2309.12053) · ALLaM (arXiv:2407.15390) · Kuwain-1.5B (Misraj, arXiv:2504.15120) · RightNow-Arabic-0.5B-Turbo (arXiv:2605.28827) · Fanar (arXiv:2501.13944) and MorphBPE (arXiv:2502.00894) · Atlas-Chat (arXiv:2409.17912) · Fast Vocabulary Transfer (Gee et al., EMNLP 2022 Industry Track) · WECHSEL (arXiv:2112.06598)

---

*Descriptions and numbers were compiled from the cited reports as of September 2026. If a figure has drifted or a claim needs a correction, open an issue or a PR: this handbook is meant to be argued with.*

**Related repos:** [LLM-Inference-Handbook](https://github.com/h9-tec/LLM-Inference-Handbook) · [LLM-Math-Handbook](https://github.com/h9-tec/LLM-Math-Handbook) · [llm-systems-engineering-roadmap](https://github.com/h9-tec/llm-systems-engineering-roadmap) · [Awesome_Arabic_NLP](https://github.com/h9-tec/Awesome_Arabic_NLP)
