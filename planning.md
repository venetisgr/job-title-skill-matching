# TalentCLEF 2025 Task B: Job Title-Based Skill Prediction
## Experiment Plan & Tracking Document

---

## Task Overview

**Goal:** Given a job title, retrieve and rank relevant skills from a fixed corpus.  
**Primary Metric:** Mean Average Precision (MAP)  
**Secondary Metrics:** MRR, P@1, P@5, P@10  
**Output Format:** TREC Run File  

---

## Model Inventory

### LLMs (Description Generation)
| ID | Model | Type | Notes |
|----|-------|------|-------|
| `qwen35` | Qwen 3.5 (capable version) | API / Local | Proven by top TalentCLEF 2025 team |
| `opus` | Claude Opus (`claude-opus-4-6`) | API | Strong instruction following |
| `gpt54` | GPT-5.4 | API | Fast, high quality |

### Bi-Encoders (Embeddings)
| ID | Model | Type | Notes |
|----|-------|------|-------|
| `qwen3emb` | `Qwen/Qwen3-Embedding-8B` | Local (GPU) | #1 MTEB multilingual leaderboard (70.58), instruction-tuned |
| `bge_m3` | `BAAI/bge-m3` | Local (GPU) | Multilingual, 8k context, very popular |
| `openai` | `text-embedding-3-large` | API | 3072 dimensions |

### Cross-Encoders (Rerankers)
| ID | Model | Type | Notes |
|----|-------|------|-------|
| `bge` | `BAAI/bge-reranker-v2-m3` | Local (GPU) | Strong open-source reranker |
| `qwen3rnk` | `Qwen/Qwen3-Reranker-4B` | Local (GPU) | Designed to pair with Qwen3-Embedding |
| `voyage` | `voyageai/rerank-2.5` | API | Strong enterprise reranker |
| `cohere` | Cohere `rerank-v4.0-pro` | API | State-of-the-art, 32k context, #2 overall on benchmarks |

---

## Phase 0: Data Exploration

- [ ] Load training data (job2skill.tsv, jobid2terms.json, skillid2terms.json)
- [ ] Load dev data (queries, corpus_elements, qrels)
- [ ] Load test data (queries, corpus_elements)
- [ ] Compute key statistics:
  - Number of unique jobs and skills in training
  - Dev/test corpus size (number of skills)
  - Average number of relevant skills per dev query
  - Overlap between training skill URIs and dev corpus skill URIs
  - Whether dev job titles appear in training job_terms
- [ ] Understand data distribution (essential vs optional skills)

---

## Phase 1: Experiments (All tracked in W&B)

**W&B Project:** `talentclef-task-b`

### Group 1: BM25 Baselines
> Index corpus skills (concatenated aliases) and query with job titles.  
> Augmented variants expand the query with LLM-generated job descriptions.

| Run ID | Method | Augmentation | Status |
|--------|--------|-------------|--------|
| `bm25_raw` | BM25 | None | [ ] |
| `bm25_aug_qwen35` | BM25 | Qwen 3.5 descriptions | [ ] |
| `bm25_aug_opus` | BM25 | Claude Opus descriptions | [ ] |
| `bm25_aug_gpt54` | BM25 | GPT-5.4 descriptions | [ ] |

**4 runs**

---

### Group 2: Bi-Encoder Zero-Shot (No Augmentation)
> Encode job titles and skill aliases independently, rank by cosine similarity.  
> No fine-tuning — pure pretrained performance.

| Run ID | Model | Augmentation | Status |
|--------|-------|-------------|--------|
| `bienc_raw_qwen3emb` | Qwen3-Embedding-8B | None | [ ] |
| `bienc_raw_bge_m3` | BGE-M3 | None | [ ] |
| `bienc_raw_openai` | text-embedding-3-large | None | [ ] |

**3 runs**

---

### Group 3: Bi-Encoder Zero-Shot + Augmentation
> Same as Group 2 but with LLM-generated descriptions appended to inputs.  
> Uses the best-performing LLM from Group 1 results.  
> Job input: "Job: {title} [SEP] Description: {LLM description}"  
> Skill input: "Skill: {name} [SEP] Description: {ESCO description}"

| Run ID | Model | Augmentation | Status |
|--------|-------|-------------|--------|
| `bienc_aug_qwen3emb` | Qwen3-Embedding-8B | Best LLM | [ ] |
| `bienc_aug_bge_m3` | BGE-M3 | Best LLM | [ ] |
| `bienc_aug_openai` | text-embedding-3-large | Best LLM | [ ] |

**3 runs**

---

### Group 4: Cross-Encoder Full Scoring (No Augmentation)
> Score ALL (job, skill) pairs directly. Quadratic complexity.  
> ~200 queries × ~1200 skills = ~240,000 forward passes per model.

| Run ID | Model | Augmentation | Status |
|--------|-------|-------------|--------|
| `crossenc_raw_bge` | bge-reranker-v2-m3 | None | [ ] |
| `crossenc_raw_qwen3rnk` | Qwen3-Reranker-4B | None | [ ] |
| `crossenc_raw_voyage` | voyage rerank-2.5 | None | [ ] |
| `crossenc_raw_cohere` | cohere rerank-v4.0-pro | None | [ ] |

**4 runs**

---

### Group 5: Cross-Encoder Full Scoring + Augmentation
> Same as Group 4 but with LLM descriptions appended to job titles.  
> Uses best LLM from Group 1 results.

| Run ID | Model | Augmentation | Status |
|--------|-------|-------------|--------|
| `crossenc_aug_bge` | bge-reranker-v2-m3 | Best LLM | [ ] |
| `crossenc_aug_qwen3rnk` | Qwen3-Reranker-4B | Best LLM | [ ] |
| `crossenc_aug_voyage` | voyage rerank-2.5 | Best LLM | [ ] |
| `crossenc_aug_cohere` | cohere rerank-v4.0-pro | Best LLM | [ ] |

**4 runs**

---

### Group 6: Cross-Encoder Rerank (Top-100 from Best Bi-Encoder)
> Use the best bi-encoder from Groups 2/3 to retrieve top-100 candidates.  
> Then rerank those 100 with each cross-encoder.  
> ~200 queries × 100 candidates = ~20,000 pairs per model.

| Run ID | Reranker | First Stage | Augmentation | Status |
|--------|----------|------------|-------------|--------|
| `rerank_bienc_raw_bge` | bge-reranker-v2-m3 | Best bi-enc (raw) | None | [ ] |
| `rerank_bienc_raw_qwen3rnk` | Qwen3-Reranker-4B | Best bi-enc (raw) | None | [ ] |
| `rerank_bienc_raw_voyage` | voyage rerank-2.5 | Best bi-enc (raw) | None | [ ] |
| `rerank_bienc_raw_cohere` | cohere rerank-v4.0-pro | Best bi-enc (raw) | None | [ ] |
| `rerank_bienc_aug_bge` | bge-reranker-v2-m3 | Best bi-enc (aug) | Best LLM | [ ] |
| `rerank_bienc_aug_qwen3rnk` | Qwen3-Reranker-4B | Best bi-enc (aug) | Best LLM | [ ] |
| `rerank_bienc_aug_voyage` | voyage rerank-2.5 | Best bi-enc (aug) | Best LLM | [ ] |
| `rerank_bienc_aug_cohere` | cohere rerank-v4.0-pro | Best bi-enc (aug) | Best LLM | [ ] |

**8 runs**

---

### Group 7: Cross-Encoder Rerank (Top-100 from BM25)
> Use BM25 (raw or augmented) to retrieve top-100 candidates.  
> Then rerank with each cross-encoder.

| Run ID | Reranker | First Stage | Augmentation | Status |
|--------|----------|------------|-------------|--------|
| `rerank_bm25raw_bge` | bge-reranker-v2-m3 | BM25 raw | None | [ ] |
| `rerank_bm25raw_qwen3rnk` | Qwen3-Reranker-4B | BM25 raw | None | [ ] |
| `rerank_bm25raw_voyage` | voyage rerank-2.5 | BM25 raw | None | [ ] |
| `rerank_bm25raw_cohere` | cohere rerank-v4.0-pro | BM25 raw | None | [ ] |
| `rerank_bm25aug_bge` | bge-reranker-v2-m3 | BM25 aug | Best LLM | [ ] |
| `rerank_bm25aug_qwen3rnk` | Qwen3-Reranker-4B | BM25 aug | Best LLM | [ ] |
| `rerank_bm25aug_voyage` | voyage rerank-2.5 | BM25 aug | Best LLM | [ ] |
| `rerank_bm25aug_cohere` | cohere rerank-v4.0-pro | BM25 aug | Best LLM | [ ] |

**8 runs**

---

### Group 8: Fusion (BM25 + Bi-Encoder via Reciprocal Rank Fusion)
> Combine rankings from BM25 and bi-encoder using RRF.  
> Tune RRF `k` parameter on dev set.

| Run ID | Components | Augmentation | Status |
|--------|-----------|-------------|--------|
| `fusion_raw` | Best BM25 raw + Best bi-enc raw | None | [ ] |
| `fusion_aug` | Best BM25 aug + Best bi-enc aug | Best LLM | [ ] |
| `fusion_mixed` | Best BM25 aug + Best bi-enc raw (and vice versa) | Mixed | [ ] |

**3 runs**

---

### Group 9: Fusion + Cross-Encoder Rerank
> Take fused rankings from Group 8, rerank top-100 with best cross-encoder.  
> This is the full pipeline: retrieve → fuse → rerank.

| Run ID | First Stage | Reranker | Augmentation | Status |
|--------|------------|---------|-------------|--------|
| `fusion_rerank_raw` | fusion_raw | Best cross-encoder | None | [ ] |
| `fusion_rerank_aug` | fusion_aug | Best cross-encoder | Best LLM | [ ] |

**2 runs**

---

## Run Summary

| Group | Description | Runs |
|-------|------------|------|
| 1 | BM25 Baselines | 4 |
| 2 | Bi-Encoder Raw | 3 |
| 3 | Bi-Encoder + Augmentation | 3 |
| 4 | Cross-Encoder Full Raw | 4 |
| 5 | Cross-Encoder Full + Augmentation | 4 |
| 6 | Rerank (Bi-Encoder → Cross-Encoder) | 8 |
| 7 | Rerank (BM25 → Cross-Encoder) | 8 |
| 8 | Fusion (BM25 + Bi-Encoder) | 3 |
| 9 | Fusion + Rerank | 2 |
| **Total** | | **39** |

---

## Execution Order (Wave-Based)

### Wave 1 — Set Baselines (~1 hour)
- [ ] Group 1: `bm25_raw`
- [ ] Group 2: All 3 bi-encoder raw runs
- **Purpose:** Establish floor for both retrieval families

### Wave 2 — LLM Descriptions + Augmented Retrieval (~3-4 hours)
- [ ] Generate descriptions with Qwen 3.5, Claude Opus, GPT-5.4
- [ ] Group 1: `bm25_aug_qwen35`, `bm25_aug_opus`, `bm25_aug_gpt54`
- [ ] Group 3: All 3 bi-encoder augmented runs (using best LLM from above)
- **Purpose:** Identify best LLM for descriptions, measure augmentation impact

### Wave 3 — Cross-Encoder Experiments (~2-3 hours)
- [ ] Group 4: All 4 cross-encoder full scoring raw runs
- [ ] Group 5: All 4 cross-encoder full scoring augmented runs
- [ ] Group 6: All 8 rerank-from-bi-encoder runs
- [ ] Group 7: All 8 rerank-from-BM25 runs
- **Purpose:** Find best reranker, measure reranking lift

### Wave 4 — Fusion & Final Combinations (~1 hour)
- [ ] Group 8: All 3 fusion runs (tune RRF k)
- [ ] Group 9: Both fusion + rerank runs
- **Purpose:** Combine best of each family for maximum MAP

---

## W&B Tracking Schema

### Config (logged per run)
```python
wandb.config.update({
    "retrieval_method": "bm25|bi-encoder|cross-encoder|fusion",
    "model_name": "exact model string",
    "augmentation_llm": "none|qwen35|opus|gpt54",
    "augmentation_target": "none|job|skill|both",
    "reranker": "none|bge|qwen3rnk|voyage|cohere",
    "rerank_top_k": 100,
    "fusion_method": "none|rrf",
    "fusion_k": 60,
    "group": "group_1|group_2|...|group_9",
})
```

### Metrics (logged per run)
```python
wandb.log({
    "MAP": float,
    "MRR": float,
    "P@1": float,
    "P@5": float,
    "P@10": float,
    "inference_time_seconds": float,
    "pairs_scored_per_second": float,
})
```

### Artifacts (logged per run)
- TREC run file (for submission and per-query inspection)
- Per-query AP breakdown table (identifies hardest job titles)

---

## Performance Targets (Based on TalentCLEF 2025 Results)

| Level | MAP | What It Takes |
|-------|-----|---------------|
| Baseline (zero-shot, no desc) | ~0.15–0.22 | Off-the-shelf encoder |
| Mid-tier (augmented zero-shot) | ~0.25–0.32 | LLM descriptions + good encoder |
| Competitive | ~0.30–0.35 | Fine-tuned bi-encoder + LLM descriptions |
| Top tier | ~0.36+ | Large model (7B+) + LoRA + advanced loss |

---

## TODO LATER: Advanced Techniques

### Fine-Tuned Bi-Encoders
- [ ] Fine-tune Qwen3-Embedding-8B on training pairs with MNRL loss
- [ ] Fine-tune BGE-M3 on training pairs with MNRL loss
- [ ] Experiment with GISTEmbed loss (used by top pjmathematician team)
- [ ] Training setup: batch size 8, grad accum 16, lr 1e-5, 20 epochs
- [ ] In-batch negatives with one job per batch
- [ ] Hard negative mining: skills from similar but different jobs

### Fine-Tuned Cross-Encoder Rerankers
- [ ] Fine-tune bge-reranker-v2-m3 on training pairs
- [ ] Fine-tune Qwen3-Reranker-4B on training pairs
- [ ] Use cross-entropy loss with positive/negative skill labels

### Classification-Based Approach
- [ ] Treat as binary classification: (job, skill) → relevant/not relevant
- [ ] NLPnorth found this approach excelled for Task B
- [ ] Multilingual encoder + classification head
- [ ] Fine-tune with cross-entropy loss on training pairs

### Ensemble Methods
- [ ] Ensemble multiple bi-encoder backbones (average scores)
- [ ] Ensemble multiple cross-encoders
- [ ] Learned ensemble weights on dev set
- [ ] Stack: use model scores as features for a meta-learner

### Advanced Data Augmentation
- [ ] Use ESCO descriptions for skills (download skills_en.csv)
- [ ] Use ESCO descriptions for jobs (where available, proxy match for others)
- [ ] Generate descriptions per alias (not per ID) for diversity
- [ ] Try different prompt templates for LLM description generation
- [ ] Generate skill descriptions with LLMs too (not just jobs)

### Query Expansion & Knowledge Injection
- [ ] Match test job titles to nearest ESCO jobs via embedding similarity
- [ ] Pull associated skills from job2skill.tsv and boost matching corpus skills
- [ ] Use ESCO taxonomy structure (skill groups, broader/narrower relations)
- [ ] Graph-based skill propagation from similar jobs

### Architectural Experiments
- [ ] LoRA fine-tuning for larger models (GTE-Qwen2-7B)
- [ ] Decoder-based embedding models (showed slight advantage in TalentCLEF 2025)
- [ ] ColBERT-style late interaction (token-level matching)
- [ ] Listwise ranking loss instead of pairwise

### Evaluation & Analysis
- [ ] Per-category analysis (which ESCO major groups are hardest?)
- [ ] Error analysis on worst-performing queries
- [ ] Ablation study on description quality vs MAP
- [ ] Statistical significance tests between top systems

---

## Key References

- **Reference repo:** https://github.com/mawhab/Job-Skill-Matching (MAP 0.345 test)
- **Working notes paper:** https://ceur-ws.org/Vol-4038/paper_363.pdf
- **TalentCLEF overview:** https://arxiv.org/abs/2507.13275
- **Task B results:** https://talentclef.github.io/talentclef/docs/talentclef-2025/results/task_b_results
- **NLPnorth paper:** https://arxiv.org/html/2506.19058
- **pjmathematician paper:** https://ceur-ws.org/Vol-4038/paper_375.pdf
- **TalentCLEF data:** https://doi.org/10.5281/zenodo.14002665
- **Evaluation script:** https://github.com/TalentCLEF/talentclef25_evaluation_script
- **Qwen3-Embedding paper:** https://arxiv.org/abs/2506.05176
