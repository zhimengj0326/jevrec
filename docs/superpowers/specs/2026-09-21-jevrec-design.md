# JevRec — A System-One Calibrated Scorer for Recommendation Ranking

**Date**: 2026-09-21
**Status**: Design confirmed, pending conversion into an implementation plan
**Deliverable**: **A paper** (target venue, see §10) + reproducible code
**Resources**: AutoDL, RTX PRO 6000 96GB — 1 card for W0–W1, 2 cards from W2 onward (see §8.1)
**Duration**: 4–4.5 weeks
**Working mode**: Gate-based autonomy (see §11)

---

## 1. Background and motivation

### 1.1 What Jev is

TypeSafe AI released **Jev** in September 2026, calling it a "System One Model." The publicly disclosed properties are:

- **Non-autoregressive**: a "parallel sampler" produces all outputs in a single forward pass
- **Output space enumerated in advance**: the caller declares the fields and each field's allowed values, so a type error is "mathematically impossible" — this is the source of zero hallucination
- **Independence across questions**: multiple questions in one call are evaluated in parallel; one answer never becomes context for another
- **Calibrated output**: every answer comes with a calibrated probability + confidence score
- **Trained via RLCD** (Reinforcement Learning for Calibrated Decisions), details undisclosed
- Latency 70–500ms; input $0.042/MTok; **output tokens free**; cardinality ≤ 255

Community consensus (latent.space, seangoedecke): the most credible explanation is "a non-autoregressive classifier over a bounded output space," billed per-inference rather than per-token. Several open-source reproductions already exist (`mini-jev`, `Decider`, `Laya`, `jevmlx`, `NanoJev`), but the systemonemodels.org roundup states plainly: **"RLCD has never been described in enough detail to copy."**

### 1.2 Why this is isomorphic to recommendation ranking

Jev's calling pattern is: **one shared context + N independently parallel questions + one calibrated probability per question**.

The shape of recommendation fine-ranking is: **one user context + N candidate items + one pCTR per item**.

This isn't an analogy — it is the same computation graph. The user history is encoded exactly once (a shared prefix), and N candidates are scored in parallel — this is precisely the only shape in which fine-ranking can plausibly run an LLM within a realistic compute budget.

### 1.3 Where this project sits

Of Jev's three properties, two can be reproduced on open Qwen models; the third cannot:

| Jev property | Reproducible? | This project's approach |
|---|---|---|
| Output space enumerated in advance → zero hallucination | ✅ | Drop the LM head, swap in a decision head |
| N questions share context, evaluated independently in parallel | ✅ | Block-diagonal attention mask + shared prefix |
| RLCD → calibrated probability | ❌ Not disclosed enough to reproduce | **This project's contribution** |

**Core claim**: the gap RLCD leaves behind happens to be exactly the problem recommendation research has worked on for over a decade (negative-sampling bias correction, IPS/DR, Platt/isotonic recalibration). This project systematically brings that toolkit to bear on a System-One-style scorer, and measures which methods actually work on an **unbiased random-exposure set**.

**Why this angle is unclaimed**: existing LLM-ranker literature reports only ranking metrics like NDCG/HR, which are completely immune to calibration. But in industry, ECPM = pCTR × bid — ad bidding consumes the calibrated probability directly. The niche of "a calibrated LLM scorer" has no occupant.

---

## 2. Scope and non-goals

### In scope
- A non-autoregressive ranking model, based on Qwen3-0.6B/1.7B-Base, that scores N candidates in a single forward pass
- A full evaluation across three metric families: ranking quality, calibration quality, and efficiency
- A quantitative characterization of the tension between ranking and calibration
- **Paper-grade deliverables**: a related-work survey, a complete baseline matrix, significance testing (see §10.3)

### Explicit non-goals
- **No retrieval.** This project only does fine-ranking; the first-stage retriever is fixed.
- **No training a non-autoregressive architecture from scratch** (the `NanoJev` route). Discarding Qwen's text-understanding ability would be disarming ourselves, and it's infeasible on a single GPU in 4 weeks.
- **No reproducing RLCD.** There isn't enough information, and any claim of reproduction would be dishonest.
- **No chasing SOTA NDCG.** The main line is calibration.

---

## 3. Data pipeline

### 3.1 Unified sample format

```python
{
  "context":    [item_1, ..., item_L],   # user's most recent L=20 interactions, shared prefix
  "candidates": [cand_1, ..., cand_N],   # N=20 for training (1 positive, 19 negatives), N=100 for evaluation
  "labels":     [...],                   # Amazon: 0/1; KuaiRand: one column per target, M targets
}
```

At training time, each list is 1 positive + 19 negatives. The gap between this sampling ratio and the real CTR is exactly what the L1 base-rate correction in §5.5 must handle.

### 3.2 Amazon Reviews 2023 — responsible for "ranking quality + text understanding"

- Category: Beauty_and_Personal_Care / Sports_and_Outdoors / Toys_and_Games
- 5-core filtering, leave-one-out split (follows the TIGER/P5 convention for comparability)
- Item text: `title + category + brand + price + description[:100]`, capped at ~48 tokens/item
- Context prefix ≈ 20 × 48 ≈ **960 tokens, encoded only once for the entire pass**
- Negatives use **popularity-based sampling** (non-uniform), closer to the real exposure distribution

**Known limitation (explicitly acknowledged)**: Amazon has only implicit positive feedback; the sampled negatives don't represent real exposure, so **there is no ground truth for P(click)**.

**Therefore**: on Amazon we **do not claim absolute calibration**, only **listwise conditional calibration** — in an N-choose-1 setting, the model's softmax probability should match the empirical frequency with which "this candidate is indeed the next interaction." This quantity is well-defined and measurable, and it matches how a reranker is actually used.

**All claims of absolute calibration rest entirely on KuaiRand** (this trade-off has been confirmed with the project owner).

### 3.3 KuaiRand — responsible for "true calibration + multi-target"

Uses KuaiRand-Pure / KuaiRand-1K. It provides two things unavailable elsewhere:

1. **Random-exposure samples** (the `is_random` flag): an unbiased ground truth for P(click | user, item). This is the gold standard for calibration evaluation.
   - Design: **train on biased normal-exposure logs, evaluate calibration on the unbiased random-exposure set**. This by itself constitutes a core experiment — which debiasing methods actually hold up under distribution shift.
2. **Multi-target feedback** (click / like / follow / forward / watch-ratio): directly isomorphic to Jev's "fill in multiple independent fields in one call."

**Known limitation**: item-side text is almost nonexistent — only tag / video_type / duration / music_id are available. A **feature-as-text template** stitches these into a short sentence (e.g., "a 32-second cooking short video, tag: home cooking"). This will weaken the LLM's relative advantage — **this is expected**; KuaiRand is not responsible for demonstrating text-understanding ability.

### 3.4 Division of labor, in summary

**Amazon proves ranking, KuaiRand proves calibration.**

---

## 4. Model architecture: JevRec

Backbone: **Qwen3-0.6B-Base** (primary) + **Qwen3-1.7B-Base** (scaling comparison).
Base rather than Instruct: this task doesn't need instruction-following, it needs representational capacity, and we use full-parameter fine-tuning.

### 4.1 Scoring N candidates in a single forward pass

Input layout:

```
[BOS][user context (L_ctx)][SEP][cand_1][SEP][cand_2] ... [SEP][cand_N]
```

The attention mask is "block-diagonal + shared prefix": each candidate segment **can see the full context, can see within its own segment (causally), and cannot see any other candidate**.

Implemented with PyTorch FlexAttention:

```python
def mask_mod(b, h, q_idx, kv_idx):
    q_seg, kv_seg = seg_id[b, q_idx], seg_id[b, kv_idx]
    return ((kv_seg == 0) | (q_seg == kv_seg)) & (kv_idx <= q_idx)
    #        sees context    or   same segment       and causal
# seg_id: 0 = context, i = candidate i
```

This is the most faithful approximation, achievable on an open-source stack, of Jev's "N questions independent and parallel, one answer never becomes context for another."

**Source of the efficiency gain**: self-attention over the context is computed once rather than N times.

### 4.2 Two correctness traps that must be handled

**① Position ids must be reset per segment.**
Context uses `0..L_ctx-1`; **every candidate segment's position ids restart from `L_ctx`**.

Without the reset, candidate order leaks into the score, and the model learns "earlier in the order is more likely to be clicked" — **every experimental number would be contaminated**. With the reset, the output is strictly invariant to candidate permutation.

This item serves double duty: it is both a required ablation (measured order-sensitivity) and a unit test (see §7.1).

**② Must confirm the backbone has no sliding-window attention layers.**
If some layers use SWA, then with a sufficiently long context, later candidates won't be able to see the full context, breaking the semantics of the block-diagonal mask.

The Qwen3 dense series is **believed to** use full attention throughout, but **this design has not verified it**. The first W0 task is to check the config to confirm this — the 1.7B model especially needs checking.

### 4.3 Output heads

Take the **last-token** hidden state `h_i ∈ R^d` of each candidate segment and feed it into an output head. The head's form **depends on the label type**:

| Field | Label type | Head | Corresponding Jev type |
|---|---|---|---|
| click / like / follow | Binary | **Scalar sigmoid** | `Choice` |
| **watch ratio** (`play_time_ms / duration_ms`) | Continuous [0, ~1.2] | **K-bin distributional head** | `Score` |

(The `forward` signal is too sparse and is excluded from the main experiments. The main experiments fix on four fields: click / like / follow / watch-ratio.)

**Why binary fields don't use a distributional head** (this was corrected during the design process, and the reasoning needs to be on record):

For a binary label, the target can only be one-hot at bin 0 or bin K-1. The CE-optimal solution is exactly the conditional distribution of the target:

```
p[0] = 1 - P(click),  p[K-1] = P(click),  the K-2 bins in between get zero mass
```

What gets learned is inevitably bimodal at the two ends. So the expected value = P(click) (identical to sigmoid), and the entropy = the binary entropy H(P(click)) (a deterministic function of P, maximized at P=0.5). **The entropy carries no information beyond the sigmoid** — it's simply restating "the prediction is close to 0.5" as "uncertain," which is a tautology of aleatoric uncertainty, not confidence. The K-2 middle bins are dead parameters.

**Why watch-ratio uses a distributional head**: viewing behavior is naturally bimodal (users either swipe away within seconds or watch to completion). The mean is a misleading summary; the distribution can express both "35% probability of swiping away, 45% probability of watching to the end" at once. Here the shape of the distribution carries real information, so confidence is well-defined.

Trained with **HL-Gauss** (smearing continuous targets into a Gaussian soft label and running CE; Farebrother et al. 2024 show this is reliably more stable than direct regression).

**Choice of K**: **default K=64**. A separate run at **K=255** aligns with Jev's official `cardinality ≤ 255` — but to be clear, 255 is a narrative-alignment choice rather than a technical requirement; K is an ordinary hyperparameter, and in practice 64 is usually sufficient. Comparing the two runs is itself a low-cost ablation.

### 4.4 Confidence for binary fields

Since entropy is redundant, we switch to a real mechanism: **after main training completes, freeze the backbone and train 3 lightweight MLP heads in parallel on top of it (different random init + different data order), using inter-head disagreement as epistemic confidence**.

The training cost of the 3 MLP heads is negligible, but what it produces is "how unsure the model is of itself," not "the prediction is close to 0.5."

Ablation: entropy vs. inter-head disagreement — which one actually predicts the error rate.

---

## 5. Training objective and calibration methods (the main contribution)

### 5.1 Core tension

Training data comes from **exposure logs already filtered by an online recommendation system**, while calibration is evaluated on a **random-exposure set**. The marginal CTR between the two differs by an order of magnitude — an online system only shows what it thinks is good, so the average CTR in the logs is far higher than in random traffic.

**A model that is perfectly calibrated on biased logs will necessarily overestimate systematically once moved to random exposure.**

### 5.2 A structural fact: listwise loss provides no calibration signal whatsoever

```
L_rank = -log( exp(s_pos) / Σ_j exp(s_j) )
```

Adding the same constant C to every `s_j` leaves this loss completely unchanged. A ranking loss constrains only the **relative order** of scores and knows nothing about their **absolute level**. A model that maxes out NDCG can have its "probabilities" shifted by a factor of ten across the board and the loss won't notice.

**Corollary**: calibration must come from pointwise supervision, ranking must come from listwise supervision, and a genuine tension exists between the two. **Characterizing this tension is the central thread of this project.**

### 5.3 Training recipe

```
L = L_rank(listwise, N candidates)  +  λ · L_cal(pointwise BCE, with bias correction)
```

**How the architecture and the loss mesh together**: the N candidates are already produced within the same forward pass — all scores come out at once, so the listwise softmax requires no additional computation.

**λ is the single most important knob in the whole project.** Sweeping λ to obtain the **ranking-quality vs. calibration-quality Pareto curve** is the headline result of the report.

### 5.4 Single head vs. dual head: an explicit design decision

**Decision: use a single shared score + a combined loss.**

There is a tempting alternative — open two heads, one governing order and one governing the numeric value, which would make the λ conflict disappear. **This design explicitly rejects that option**, for the following reason:

What Jev promises, and what ad systems need, is **one and the same number that is both correctly ordered and correctly calibrated**. ECPM = pCTR × bid — you cannot rank with one score and bid with another. A dual-head design hides the tension rather than resolving it.

The dual-head design is **retained as an ablation**, to quantify "how much can be gained for free by dodging this tension."

### 5.5 A four-rung ladder of debiasing

| Level | Method | Cost | Notes |
|---|---|---|---|
| L0 | No treatment (naive BCE) | — | Control |
| L1 | **Negative-sampling base-rate correction** (Elkan 2001): `logit -= log(π_train/(1-π_train))`, then add back the target base rate | 1 scalar | **Not optional** |
| L2 | Clipped-IPS / **Doubly Robust** (Saito et al.): reweight logged samples with a propensity model | Requires training a propensity model, high variance | |
| L3 | **Post-hoc recalibration** (temperature / Platt / isotonic) on a small sample of random exposures | Lowest | Usually the most effective |

**L1 is not optional.** Training uses 1 positive to 19 negatives while the real CTR might be 2%; without a base-rate correction, predicted values will be wildly too high, invalidating all subsequent calibration work.

### 5.6 Multi-target

Four heads — click / like / follow (sigmoid) + watch-ratio (distributional head) — share a backbone.

Jev claims that "each question is evaluated independently and in parallel." **This claim deserves to be tested**: does joint multi-task training hurt single-task calibration? Four independent single-task models serve as the control, quantifying the gap.

---

## 6. Evaluation

### 6.1 Methodological precondition (determines the report's credibility)

**All baselines must go through the same calibration ladder (L1 + L3).**

If only JevRec gets calibrated while baselines run raw, the conclusion isn't "our model calibrates better" — it's "we put in the effort and they didn't." That conclusion carries no information and is transparently hollow. Adding isotonic regression to DIN can improve ECE just as dramatically, and this must be reported honestly.

The real question is: **after every method has been given equal calibration effort, how much of an edge does JevRec retain?** The answer might be "not much" — and that too would be an honest and valuable result.

### 6.2 Baseline matrix

| Group | Model | Role |
|---|---|---|
| ID-based | **DIN** | Industry-standard fine-ranker, the main KuaiRand competitor |
| | **SASRec** (used as a scorer) | Amazon sequential baseline |
| | **LightGBM / DeepFM** (tabular features) | What's actually running in production, **the toughest competitor** |
| Text-based | **ModernBERT cross-encoder** | The most recent non-LLM text baseline (the `Laya` route) |
| | **Frozen Qwen3-1.7B reading out logits** | "Zero-training Jev" lower bound |
| | **Qwen3-1.7B autoregressive LLM-as-ranker** | **System Two control group**, cost reference point for the Pareto figure |

**On the choice of the System Two control group**: this project uses only two weight sizes, Qwen3-0.6B and 1.7B, and does not bring in 4B.

This is not merely about saving resources — it also makes the comparison cleaner. If Qwen3-4B autoregressive were compared against JevRec-0.6B, both the parameter count and the decoding method would change **simultaneously**, so a win couldn't be attributed to either factor. Instead, using **Qwen3-1.7B autoregressive vs. JevRec-1.7B**, both share the same backbone, the same parameter count, and the same training data — **the only difference is autoregressive decoding vs. single-forward-pass scoring** — which is what actually isolates the architectural claim.

The cost (must go into Limitations): this paper does not verify whether this conclusion holds at larger scale (≥4B).
| Ours | **JevRec-0.6B / 1.7B** | |

### 6.3 Evaluation protocol

**Amazon (ranking)**: does not use the "1 positive + 99 random negatives" sampled metric — Krichene & Rendle 2020 have already shown that sampled metrics are inconsistent with full ranking. Instead, use a **realistic reranking scenario**: a **SASRec trained on the same training set serves as a fixed first-stage retriever**, retrieving the top-100 from the full item pool; every reranking method scores and orders only these 100.

**The retriever is fixed throughout and identical for every method**, ensuring the comparison isolates reranking ability itself. Note that SASRec also appears in the baseline list (§6.2) — the two roles are different: as a retriever it produces the candidate set, as a baseline it re-scores that same candidate set. Both use the same weights, so this does not constitute unfairness.

**KuaiRand (the calibration main line)**:
- Split train/test **by time**, not randomly (to avoid look-ahead leakage)
- Train on normal exposure traffic; **calibration is evaluated on the held-out random-exposure set**
- Ranking metrics (GAUC) are reported **on both** normal and random traffic — the gap between the two is itself a measure of the strength of the distribution shift

**Efficiency measurement**: A100, bf16, N=100 candidates. Report end-to-end latency at batch=1 + "candidates scored per second" at maximum throughput. Must include warmup, CUDA sync, and take the median over multiple runs. Three controls: ① scoring candidates one at a time across N runs (quantifies the gain from the block-diagonal mask) ② ModernBERT ③ Qwen3-1.7B autoregressive.

**Statistical rigor**: GAUC/NDCG get bootstrap confidence intervals computed per user; **ECE binning is noisy, so it must be bootstrapped**; the main model is run with 3 seeds.

### 6.4 Metrics

**Ranking**: NDCG@10, GAUC (per-user AUC)

**Calibration** (all computed on the held-out random-exposure set):
- **ECE** (15 equal-frequency bins) + **Adaptive ECE**
- **Calibration / refinement decomposition of the Brier score** — separates "well-calibrated" from "well-discriminated"
- **Calibration slope & intercept**: logistic regression of y on the logit, perfect at slope=1, intercept=0. Distinguishes "overall shift" from "overconfidence" better than ECE
- **PCOC** = `mean(pred) / mean(label)` — the metric actually used in ad systems

**Efficiency**: end-to-end latency, candidates scored per second

### 6.5 Ablations

**Must do:**
- **λ sweep → ranking-calibration Pareto curve** (headline result)
- **Position-id reset on/off** → candidate order sensitivity
- Debiasing ladder L0 / L1 / L2 / L3
- Single head vs. dual head

**Do if time allows:**
- **Block-diagonal mask vs. mutually visible candidates**: mutual visibility among candidates = listwise context, **may improve ranking**, but breaks Jev's independence promise and permutation invariance. Worth honestly quantifying this tension
- Multi-task vs. single-task (tests Jev's claim that "each question is independent")
- Entropy vs. 3-head disagreement — which one actually predicts errors
- Throughput and quality scaling at N = 20/50/100

### 6.6 Appendix content

**"How many random-exposure samples are needed to push ECE down to a target level"** (sweep 100 / 1k / 10k / 100k for L3 post-hoc recalibration).

This result has practical value for industry (random traffic must be paid for with online user experience, so "how large a random bucket to keep" is a real product decision), but it is more engineering than a model contribution, so **the project owner has decided to place it in the appendix**.

---

## 7. Engineering and verification

### 7.1 Key unit test

**Candidate permutation-invariance test (blocking)**: apply a random permutation to the same set of candidates; the model's output scores must be identical element-wise (within numerical tolerance).

If this test fails, it means the position-id reset or the attention mask is wrong, and **all subsequent experimental data is invalid**. Must pass by the end of W0.

### 7.2 Fallback implementation

FlexAttention's custom mask may not hit the fast path, and could end up slower than a naive implementation. **W0 benchmarks both paths simultaneously**:

- **A**: FlexAttention + block-diagonal mask (single forward pass)
- **B**: encode the context once, cache its KV, and reuse that prefix by batching candidates as short sequences

B is simpler to implement, and **may be faster**. The two are mathematically equivalent; pick whichever performs better.

---

## 8. Compute budget and timeline

### 8.1 Compute budget and GPU rental plan

**VRAM is not the bottleneck.** State memory for full-parameter fine-tuning: Qwen3-0.6B ≈ 10 GB, 1.7B ≈ 27 GB — 96 GB leaves ample headroom (the project only uses the 0.6B/1.7B weights). The real benefit of the 96 GB card is limited to: larger batch sizes, no need for gradient checkpointing, roughly a 30% speed gain.

**Compute is the bottleneck.** Tokens per training sample: KuaiRand ≈ 960 (context 20×24 + candidates 20×24), Amazon ≈ 1900. Estimating FLOPs as `6 × parameter count × token count`, with the card at ~125 TFLOPS BF16 and MFU 35–40%:

| Experiment unit | Cost |
|---|---|
| One full KuaiRand 0.6B training run | ~13 GPU-h |
| One Amazon 0.6B run | ~28 GPU-h |
| 1.7B | ×3 |

| Item | GPU-h |
|---|---|
| λ sweep, 6 points (KuaiRand, 0.6B) **← headline result** | 78 |
| 3 seeds at the optimal λ | 26 |
| ~8 ablation groups | 104 |
| 1.7B scaling, 3 runs | 120 |
| All baselines (incl. Qwen3-1.7B autoregressive inference) | 30 |
| Amazon main experiments | 140 |
| Subtotal | ~500 |
| Reruns/debugging/eval overhead +30% | **~650** |

**One card for 4.5 weeks × 20 effective hours/day ≈ 630 GPU-h** — arithmetically exactly 100% utilized, with zero margin. Zero margin on an ML project means any single "ran it and then found a mislabeled field" instantly blows through the timeline.

**GPU rental plan**:
- **Rent 1 card for W0–W1 (the first ~10 days).** This phase is writing code, checking configs, tuning the mask, and running unit tests — a second card would have nothing to do; and there's a Go/No-Go at the end of W1, so if the approach doesn't pan out, 10 days of rental can be saved.
- **Add a 2nd card when the W2 λ sweep starts.** The sweep is 6 fully independent training runs, naturally parallel.
- **Rent two independent single-GPU instances rather than one dual-GPU instance.** The workload consists of mutually independent tasks, not distributed training; the RTX PRO 6000 Blackwell has no NVLink, so dual-card communication goes over PCIe, and DDP offers no benefit; single-GPU instances are easier to acquire and can be released independently. The cost is that the data disks on the two instances need to be synced separately — this can be worked around by mounting AutoDL file storage onto both instances in the same region.

**Disk reality (measured 2026-09-21)**: the data disk `/root/autodl-tmp` can only be expanded to **60G** (not the originally planned 200G), and the 30G system disk is not guaranteed to persist. AutoDL's public-dataset area has neither KuaiRand nor Amazon.

Given this, the checkpoint retention policy is upgraded to a hard constraint: **keep only one checkpoint at a time** (evaluate immediately after training → persist prediction files and metrics → delete the weights; calibration analysis consumes predictions, not weights; only the 2–3 runs that go into the paper are archived); Amazon raw files are deleted immediately after processing; preprocessing outputs store token-id arrays, not text; every disk-writing script must pass a disk guard before starting. Under this estimate, total usage is about 20G, leaving ample margin.

**Currently only 1 card has actually been rented**; the dual-card plan is deferred and to be reassessed after G1. W0–W1 only ever needed 1 card anyway; if the W2 λ sweep is still single-card, it degrades to serial execution (about 78 GPU-h ≈ 4 days), at which point it will need to be reweighed.

⚠️ **The compute budget above rests on two untested assumptions**: the card's BF16 peak throughput and the actual MFU achieved. If effective compute turns out to be only half the estimate, the conclusion flips from "1 card is just enough" to "1 card is not enough." **Day one of W0 must measure real tokens/sec and pin this number down** — this is a W0 blocking item.

### 8.2 Timeline

| Week | Content | Output / Gate |
|---|---|---|
| **W0** (3-4 days) | Unify the data pipeline; **check the Qwen3 config to confirm no SWA**; **tally the volume of KuaiRand random-exposure samples**; **measure real tokens/sec (blocking item, see §8.1)**; implement the mask (benchmark both paths A/B); **permutation-invariance test**; get the frozen-Qwen baseline running end-to-end | First number on the board + compute budget pinned down |
| **W1** | JevRec-0.6B pure-listwise training working end-to-end, ranking quality on par with SASRec/DIN | **⚠️ Go/No-Go gate** |
| **W2** | The calibration main line: pointwise + L1 + λ sweep + L3; the full ECE/PCOC/slope metric suite | Pareto curve |
| **W3** | All baselines (**each through the same calibration ladder**) + ablations + efficiency measurement + 1.7B scaling | Full matrix |
| **W4** | Follow-up experiments + significance testing + writing the paper (including related work) | Paper draft |

**Total duration**: W0 occupies the first 3-4 days of week 1, and W1 is the rest of that week plus a small overflow, so the actual total duration is **4 to 4.5 weeks**, not a strict 4. If W0's environment/data issues run over expectation, the first thing to be trimmed is the number of Amazon baselines (see the priority order below).

**Go/No-Go at the end of W1**: if JevRec's pure ranking can't beat SASRec/DIN, **stop and reassess — don't push through regardless.**

**Dataset priority (for trimming if time runs short)**: KuaiRand first (carries the main thesis), Amazon second; Amazon can be downgraded to "ranking only + a slimmed-down baseline set."

---

## 9. Risks

**R1 (highest): ranking loses to LightGBM / DIN.**
KuaiRand's item side has almost no text while tabular features are extremely rich, so ID+feature models are strong on this kind of data, and the LLM's advantage is hard to bring out. **This is not a low-probability event.**
- Mitigation: hybrid input — item text template **concatenated with a pretrained ID embedding**
- Fallback: if ranking genuinely falls behind, narrow the claim to "calibration + cost." **Note this weakens the paper — "worse ranking but better calibration" is a hard story to sell** — at that point the submission target needs to be reassessed at the G2 gate (possibly downgraded to a workshop or short paper)
- **This risk is declared at the design stage and should not first be discovered in W3**

**R2: FlexAttention performance falls short of expectations.** Mitigation in §7.2 (fallback implementation B).

**R3: KuaiRand's random-exposure sample volume is insufficient for a stable ECE estimate.** Tally this first in W0. If insufficient, use adaptive binning + bootstrap intervals and disclose this honestly in the report.

**R4: 4 weeks for two datasets is tight.** Mitigation is the dataset priority order in §8.

**R5: Qwen3 config surprises** (SWA, tokenizer, RoPE scaling). Verify in W0; a known unknown.

---

## 10. Paper structure and novelty claim

### 10.1 The claim must stand on calibration, not on Jev

Jev was released on 2026-09-16, only 5 days before this design. Being first is an advantage, but **reviewers are quite likely to read "following a closed-source product that's been live for a week" as chasing hype**, and if Jev's buzz fades, a paper anchored to it fades along with it.

**So the paper's central claim is the calibration contribution, with Jev appearing only as motivation.** Specifically:

> Existing LLM-based ranker literature reports only ranking metrics such as NDCG/HR, which are completely immune to the absolute level of the scores. But industrial ranking systems (especially advertising) consume calibrated probabilities, not rank order. This paper presents the first systematic study of the quantifiable tension between ranking quality and calibration quality in a non-autoregressive LLM scorer that produces N scores in a single forward pass, and evaluates the practical effectiveness of a four-level debiasing ladder on unbiased random-exposure data.

This claim would still hold even if Jev disappeared tomorrow.

### 10.2 Paper structure

1. **Introduction** — opens from the fact that industrial ranking consumes calibrated probabilities; Jev serves as timely evidence that "non-autoregressive scorers already exist in products"
2. **Related Work** — three lines: (a) LLM for recommendation / generative rec (the P5, TIGER, OneRec family); (b) debiasing and unbiased learning in recommendation (IPS, DR, Saito et al.); (c) neural network calibration (temperature scaling, focal loss, ECE metrology). **This paper's position: the unclaimed intersection of the three lines**
3. **Method** — §4 architecture + §5 training objective
4. **Experiments** — §6
5. **Analysis** — interpreting the λ Pareto curve; why listwise objectives and calibration are structurally in conflict
6. **Limitations** — see §10.4
7. **Appendix** — the random-sample-size curve from §6.6; full hyperparameters

### 10.3 Additional requirements for paper-grade work (relative to a technical report)

- **Significance testing**: every comparison in the main results table gets a paired bootstrap test, reporting p-values or confidence intervals; start from 3 seeds
- **Baseline completeness**: the §6.2 matrix must be run in full; omissions "due to time constraints" are not allowed without disclosure
- **Reproducibility**: code + configs + data-processing scripts released with the paper; fixed random seeds; environment recorded

### 10.4 What must go into Limitations

- Evidence for absolute calibration comes **from only one dataset, KuaiRand** (Amazon has no ground truth, only conditional calibration)
- **RLCD was not reproduced**; this paper's method is an alternative, not a replication
- KuaiRand's item-side text is sparse, so the LLM's text-understanding advantage cannot be fully demonstrated on this dataset
- Not validated on a real online system
- Scale capped at 1.7B, **whether the conclusion holds at ≥4B is not verified**

---

## 11. Working mode: gate-based autonomy

### 11.1 Scope of autonomous execution

Fully autonomous between gates: writing all the code (data pipeline, model, training, evaluation, plotting); SSH into AutoDL to launch jobs, monitor them, and collect results; analyzing results, generating LaTeX tables and figures; drafting the paper. Each autonomous stretch runs 2–7 days without human involvement.

### 11.2 Three mandatory stopping gates

| Gate | Timing | What to check |
|---|---|---|
| **G1 Correctness** | End of W0 | Whether the permutation-invariance test passes; real tokens/sec; volume of KuaiRand random-exposure samples |
| **G2 Go/No-Go** | End of W1 | Whether pure ranking beats SASRec/DIN. **Whether to switch approaches or keep tuning if it doesn't is a research judgment call, made by the project owner** |
| **G3 Main result** | End of W2 | The λ Pareto curve. Its shape determines the paper's claim ("calibration is nearly free" and "calibration costs 5 points of NDCG" are two different papers) |

### 11.3 Known failure modes under autonomous operation

**The biggest risk isn't failing to finish — it's finishing a pile of self-consistent but wrong work.**

A typical symptom: some metric suddenly looks abnormally good (e.g., ECE plunges), and nine times out of ten this comes from label leakage or eval-set contamination rather than the method actually working. Automated pipelines tend to accept results like this as discoveries.

**Mitigation (to be written into the implementation plan)**:
- The first time any new metric shows an anomalously good number, **force a leakage self-check** (check the time split, check whether features contain label-derived quantities, run once with shuffled labels to see if the metric collapses)
- The shuffle-label sanity check is a standing regression test, not an ad hoc measure
- Gate reports must list "counterintuitive results that appeared this phase and how they were investigated" — silent pass-through is not allowed

---

## 12. References

- [TypeSafe AI: Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- [latent.space: Jev — a "System One Model" that only decides/classifies/routes/scores](https://www.latent.space/p/ainews-jev-a-system-one-model-that)
- [seangoedecke: Jev means structured output is interesting again](https://www.seangoedecke.com/jev-means-structured-output-is-interesting-again/)
- [systemonemodels.org: open-source reproduction roundup](https://systemonemodels.org/examples/alternatives/)
- Elkan 2001, *The Foundations of Cost-Sensitive Learning* (base-rate correction)
- Saito et al., *Doubly Robust Estimator for Ranking Metrics with Post-Click Conversions* (DR debiasing)
- Krichene & Rendle 2020, *On Sampled Metrics for Item Recommendation* (problems with sampled metrics)
- Farebrother et al. 2024, *Stop Regressing: Training Value Functions via Classification* (HL-Gauss)
