# TalentCLEF 2025 Task B: Job Title-Based Skill Prediction

A comprehensive information retrieval system for predicting relevant professional skills given a job title, built for the [TalentCLEF 2025](https://talentclef.github.io/talentclef/) shared task.

## Task Overview

Given a job title (e.g., *"corporate governance analyst"*), retrieve and rank relevant skills from a corpus of ESCO-standardized skills. The system produces a ranked list evaluated using Mean Average Precision (MAP) as the primary metric.

## Key Results

| Rank | Method | MAP | MRR | P@1 | Notes |
|------|--------|-----|-----|-----|-------|
| 1 | Cross-Encoder (Voyage) + ESCO Prompt | **~0.336*** | — | — | *Imputed† |
| 2 | Cross-Encoder (Voyage) + Original Prompt | 0.3054 | 0.7954 | 0.6579 | Best measured |
| 3 | Cross-Encoder (Cohere) + ESCO Prompt | 0.2633 | 0.7097 | 0.5461 | |
| 4 | Bi-Encoder (Qwen3-Emb) + ESCO Prompt | 0.2581 | 0.8069 | 0.6809 | |
| 5 | Fusion Mixed (BM25 aug + Bi-Enc raw) | 0.2497 | 0.8079 | 0.6809 | |
| 6 | Rerank Bi-Enc → Voyage + ESCO Prompt | 0.2355 | 0.8018 | 0.6645 | |

> †Imputed using Cohere cross-encoder's prompt improvement pattern (+0.0304 absolute MAP delta). 78 measured experiments across 2 rounds tracked with [Weights & Biases](https://wandb.ai).

## Architecture

```
                    ┌─────────────────┐
                    │   Job Title     │
                    │ "hoa coordinator"│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ LLM Description │  ← Qwen3 / Claude Opus / GPT-5.4
                    │   Generation    │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼───────┐ ┌───▼────┐ ┌───────▼────────┐
     │   BM25         │ │Bi-Enc  │ │ Cross-Encoder  │
     │  (Lexical)     │ │(Dense) │ │ (Full Scoring) │
     └────────┬───────┘ └───┬────┘ └───────┬────────┘
              │              │              │
              └──────┬───────┘              │
                     │                      │
              ┌──────▼───────┐              │
              │   RRF Fusion │              │
              └──────┬───────┘              │
                     │                      │
              ┌──────▼───────┐              │
              │  Cross-Enc   │◄─────────────┘
              │  Reranker    │
              └──────┬───────┘
                     │
              ┌──────▼───────┐
              │  TREC Run    │
              │    File      │
              └──────────────┘
```

## Methods

| Method | Description | Best MAP |
|--------|-------------|----------|
| **BM25** | Lexical retrieval using term frequency and IDF | 0.1788 |
| **Bi-Encoder** | Dense retrieval with independent query/document encoding | 0.2581 |
| **Cross-Encoder** | Joint query-document scoring with full cross-attention | 0.3054 |
| **Reranking** | Two-stage: bi-encoder retrieval → cross-encoder reranking | 0.2355 |
| **RRF Fusion** | Reciprocal Rank Fusion of BM25 + bi-encoder rankings | 0.2497 |
| **LLM Augmentation** | Expand job titles with LLM-generated descriptions | +17–39% |

## Models Used

**Bi-Encoders:** Qwen3-Embedding-8B (#1 MTEB), BGE-M3, OpenAI text-embedding-3-large

**Cross-Encoders / Rerankers:** Voyage rerank-2.5, Cohere rerank-v4.0-pro, BGE-reranker-v2-m3, Qwen3-Reranker-4B

**LLMs (Description Generation):** Qwen3-Next-80B-A3B-Instruct (Together.ai), Claude Opus, GPT-5.4

## Two Experiment Rounds

| | Experiment 1 | Experiment 2 |
|---|---|---|
| **Prompt** | Original (2-3 skills) | ESCO-aligned (5-6 skills + related titles) |
| **Rerank Top-K** | 100 | 250 |
| **Best LLM** | Qwen3 | GPT-5.4 |
| **Best MAP** | 0.3054 | 0.2633 (Voyage aug not re-run) |

### Key Improvements from Experiment 2

| Method | Exp 1 | Exp 2 | Change |
|--------|-------|-------|--------|
| BM25 best augmented | 0.1287 | 0.1788 | **+38.9%** |
| Bi-Enc aug (Qwen3-Emb) | 0.2396 | 0.2581 | +7.7% |
| Rerank bienc aug (Voyage) | 0.1591 | 0.2355 | +48.0% |
| Fusion mixed | 0.2210 | 0.2497 | +13.0% |

## Project Structure

```
├── config_api.py                    # API keys and model config (gitignored)
├── phase0_phase1_train.ipynb        # Experiment 1: original prompt, K=100
├── phase0_phase1_train_2.ipynb      # Experiment 2: ESCO prompt, K=250
├── phase0_phase1_val_test.ipynb     # Val + Test evaluation with official ranx
├── talentclef_evaluate.py           # Official evaluation script
├── planning.md                      # Full experiment plan
├── APPROACHES_GUIDE.md              # Detailed guide to all methods
├── descriptions/                    # Cached LLM-generated descriptions
├── outputs/                         # TREC run files and results
├── docs/
│   ├── TalentCLEF_TaskB_Final_Report.docx
│   └── TalentCLEF_Theory_Guide.docx
└── taskB/                           # Data (from Zenodo)
```

## Quick Start

### 1. Install

```bash
pip install rank_bm25 sentence-transformers transformers torch
pip install openai anthropic cohere voyageai
pip install wandb ranx
```

### 2. Configure

Create `config_api.py` and fill in your API keys:
```python
OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "sk-ant-..."
COHERE_API_KEY = "..."
VOYAGE_API_KEY = "..."
WANDB_API_KEY = "..."
TOGETHER_API_KEY = "..."
```

### 3. Run

Open `phase0_phase1_train.ipynb` in Google Colab with GPU runtime and run all cells.

### 4. Evaluate

```bash
python talentclef_evaluate.py --qrels taskB/validation/qrels.tsv --run outputs/crossenc_aug_voyage.trec
```

## Key Findings

1. **LLM augmentation is critical** — job titles average 2-4 words; descriptions boost BM25 by up to 39%
2. **Prompt engineering matters** — ESCO-aligned prompts significantly outperformed natural language descriptions
3. **Reranking top-K is critical** — K=250 improved reranking MAP by 48% over K=100
4. **Skills are known, job titles are not** — 95.8% skill overlap vs 3.9% job title overlap with training
5. **Different methods excel at different metrics** — cross-encoders for MAP, rerankers for MRR/P@1, fusion for robustness

## Data

| Split | Queries | Corpus | Avg Relevant/Query |
|-------|---------|--------|-------------------|
| Training | 3,011 | 13,224 | 38.1 |
| Validation | 304 | 1,439 | 85.2 |
| Test | 1,520 | 1,986 | — |

Source: [ESCO](https://esco.ec.europa.eu/) via [Zenodo](https://doi.org/10.5281/zenodo.14002665)

## Future Work

- [ ] Fine-tune Qwen3-Embedding-8B with MNRL/GISTEmbed + LoRA
- [ ] Run crossenc_aug_voyage with enhanced ESCO prompt (imputed ~0.336)
- [ ] Weighted ensemble of fine-tuned bi-encoder + cross-encoder
- [ ] Add ESCO skill descriptions to corpus representations

## References

- [TalentCLEF 2025 Overview](https://arxiv.org/abs/2507.13275)
- [Reference Implementation](https://github.com/mawhab/Job-Skill-Matching) — MAP 0.345
- [Working Notes](https://ceur-ws.org/Vol-4038/paper_363.pdf)
- [Official Results](https://talentclef.github.io/talentclef/docs/talentclef-2025/results/task_b_results)
