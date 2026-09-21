# JevRec

A System One calibrated scorer for recommendation ranking.

A ranking model built on Qwen3-Base that is **non-autoregressive and scores N candidates
in a single forward pass**. The core research question is **the tension between ranking
quality and calibration quality**.

📄 **Full design in [`docs/superpowers/specs/2026-09-21-jevrec-design.md`](docs/superpowers/specs/2026-09-21-jevrec-design.md)**

---

## What it is, in one paragraph

The shape of ranking in production recommender systems is "one user context + N candidates +
one pCTR per candidate." This project encodes the user history only once as a shared prefix,
then scores the N candidates in parallel through a **block-diagonal attention mask** that keeps
them mutually invisible to each other, producing all scores in a single forward pass.

The core claim isn't about architecture — it's about calibration: existing LLM-ranker literature
only reports ordinal / rank-based metrics like NDCG/HR, which are entirely insensitive to the
absolute level of the scores. But industrial ranking (especially ads, where ECPM = pCTR × bid)
consumes calibrated probabilities.

---

## Repository layout

```
src/jevrec/
  data/     Data pipeline (Amazon Reviews 2023 / KuaiRand -> unified sample format)
  model/    JevRec architecture (block-diagonal mask, position reset, output head)
  train/    Training objective (combined listwise + pointwise loss, debiasing ladder)
  eval/     Ranking metrics, calibration metrics, efficiency benchmarks
configs/    Experiment configs
scripts/    Training/evaluation launch scripts
tests/      Unit tests (including the permutation invariance test -- blocking)
results/    Tables and figures for the paper
docs/       Design docs
```

## Environment

**Local**: code editing + version control, auto-synced to AutoDL.
**Remote** (AutoDL, `ssh autodl`):

- GPU: RTX PRO 6000 Blackwell Server Edition 96GB (sm_120)
- torch 2.8.0+cu128 / Python 3.12.3 / conda base

## Status

Design is finalized, pending conversion into an implementation plan. Currently before
**W0 (environment and correctness validation)**.

Blocking items for W0:
1. Measure real tokens/sec (the entire compute budget rides on this number)
2. Confirm the Qwen3 config has no sliding-window attention layers
3. Count the KuaiRand random-exposure sample size
4. **The candidate permutation invariance test must pass** -- if it fails, all subsequent
   experimental data is invalid
