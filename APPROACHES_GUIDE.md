# TalentCLEF Task B — Complete Guide to Approaches

## 1. Problem Statement

**Input:** A job title string → **Output:** Ranked list of 1,439 ESCO skills → **Metric:** MAP

**Core Challenge:** 95.8% of skills are known from training, but only 3.9% of job titles are. Average 85.2 relevant skills per query.

## 2. Data

| Statistic | Training | Validation | Test |
|-----------|----------|------------|------|
| Queries | 3,011 | 304 | 1,520 |
| Corpus Skills | 13,224 | 1,439 | 1,986 |
| Pairs | 114,699 | — | — |
| Skills/Job (mean) | 38.1 | 85.2 relevant | — |

## 3. Method 1: BM25 Lexical Retrieval

Ranks documents by exact word overlap using term frequency × inverse document frequency. Each skill = concatenated aliases. Query = job title (optionally expanded with LLM description).

**Best result:** MAP 0.179 (BM25 + GPT-5.4 ESCO-aligned descriptions)

## 4. Method 2: Bi-Encoder Dense Retrieval

Encodes query and document independently into dense vectors via a shared Transformer. Relevance = cosine similarity. Corpus embeddings pre-computed once.

| Model | Raw MAP | Best Aug MAP |
|-------|---------|-------------|
| **Qwen3-Embedding-8B** | **0.257** | **0.258** |
| OpenAI text-embedding-3-large | 0.204 | 0.212 |
| BGE-M3 | 0.174 | 0.187 |

## 5. Method 3: Cross-Encoder Full Scoring

Concatenates query + document as one input. Full cross-attention. Scores all ~437K pairs. Most accurate but slowest.

| Model | Raw MAP | Best Aug MAP |
|-------|---------|-------------|
| **Voyage rerank-2.5** | **0.262** | **0.336** |
| Cohere rerank-v4.0-pro | 0.233 | 0.263 |
| BGE-reranker-v2-m3 | 0.178 | 0.107 |
| Qwen3-Reranker-4B | 0.080 | 0.145 |

## 6. Method 4: LLM Description Augmentation

Expands 2-4 word job titles into rich descriptions with skills and context.

### Two Prompts Tested

**Original:** "Generate 2-3 skills in a 2-3 sentence description"

**Enhanced ESCO-aligned:** "Provide: 1) role summary, 2) 5-6 ESCO-terminology skills with examples, 3) 2-3 related job titles"

| LLM | Original Prompt (BM25) | Enhanced Prompt (BM25) |
|-----|----------------------|----------------------|
| Qwen3 | 0.129 (best) | 0.155 |
| Claude Opus | 0.114 | 0.171 |
| **GPT-5.4** | 0.128 | **0.179** (best) |

The enhanced prompt improved BM25 by 39%, flipped bi-encoder augmentation from harmful to helpful, and changed which LLM performed best.

## 7. Method 5: Cross-Encoder Reranking

Two-stage: fast retriever gets top-K → cross-encoder rescores only those K candidates.

### The Top-K Problem

| Run | K=100 MAP | K=250 MAP | Change |
|-----|-----------|-----------|--------|
| rerank_bienc_aug_voyage | 0.159 | 0.236 | **+48.0%** |

## 8. Method 6: Reciprocal Rank Fusion

Combines rankings: `RRF_score(d) = Σ 1/(60 + rank_i(d))`

| Variant | Exp 1 MAP | Exp 2 MAP |
|---------|-----------|-----------|
| **Mixed (BM25 aug + Bi-Enc raw)** | **0.221** | **0.250** |
| Augmented (both aug) | 0.209 | 0.241 |
| Raw (BM25 + Bi-Enc) | 0.208 | 0.208 |

## 9. Method 7: Fusion + Reranking

Full pipeline: BM25 aug + Bi-Enc raw → RRF → top-K → Voyage reranker.

Best: fusion_rerank_aug MAP 0.222, MRR 0.802 — finds the first relevant skill fastest.

## 10. Complete Results — Top 10

| Rank | Run | MAP | MRR | P@1 |
|------|-----|-----|-----|-----|
| 1 | crossenc_aug_voyage (ESCO prompt) | **0.336** | 0.833 | 0.720 |
| 2 | crossenc_aug_voyage (original prompt) | 0.305 | 0.795 | 0.658 |
| 3 | crossenc_aug_cohere (ESCO prompt) | 0.263 | 0.710 | 0.546 |
| 4 | bienc_aug_qwen3emb (ESCO prompt) | 0.258 | 0.807 | 0.681 |
| 5 | bienc_raw_qwen3emb | 0.257 | 0.801 | 0.668 |
| 6 | fusion_mixed (ESCO prompt) | 0.250 | 0.808 | 0.681 |
| 7 | fusion_aug (ESCO prompt) | 0.241 | 0.789 | 0.641 |
| 8 | rerank_bienc_aug_voyage | 0.236 | 0.802 | 0.665 |
| 9 | crossenc_raw_cohere | 0.233 | 0.675 | 0.510 |
| 10 | fusion_rerank_aug | 0.222 | 0.802 | 0.665 |

## 11. Lessons Learned

**What Worked:** Voyage rerank-2.5 dominated. Qwen3-Embedding instruction prefix was key. ESCO-aligned prompts. Mixed fusion. K=250 for reranking.

**What Didn't Work:** BGE-reranker with augmentation. Qwen3-Reranker-4B overall. Original prompt on Qwen3-Embedding. Reranking BM25 candidates. Claude Opus for query expansion.

## 12. Future Directions

1. Fine-tune Qwen3-Embedding-8B with MNRL loss + LoRA
2. Weighted RRF ensemble of fine-tuned bi-encoder + BM25 + Voyage
3. Add ESCO skill descriptions to corpus (free)
4. Classification approach for binary job-skill relevance
