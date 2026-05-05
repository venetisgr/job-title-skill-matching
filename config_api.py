# ============================================================
# config_api.py — TEMPLATE
# ============================================================
# NEVER commit config_api.py to version control.
# ============================================================

# API Keys
OPENAI_API_KEY = ""          # https://platform.openai.com/api-keys
ANTHROPIC_API_KEY = ""       # https://console.anthropic.com/settings/keys
COHERE_API_KEY = ""          # https://dashboard.cohere.com/api-keys
VOYAGE_API_KEY = ""          # https://dash.voyageai.com/api-keys
WANDB_API_KEY = ""           # https://wandb.ai/authorize
TOGETHER_API_KEY = ""        # https://api.together.xyz/settings/api-keys

# LLM Models (Description Generation)
LLM_QWEN = "Qwen/Qwen3-Next-80B-A3B-Instruct"
LLM_OPUS = "claude-opus-4-6"
LLM_GPT = "gpt-5.4"
QWEN_BASE_URL = "https://api.together.xyz/v1"

# Bi-Encoder Models
BIENC_QWEN = "Qwen/Qwen3-Embedding-8B"
BIENC_BGE = "BAAI/bge-m3"
BIENC_OPENAI = "text-embedding-3-large"

# Cross-Encoder / Reranker Models
CROSSENC_BGE = "BAAI/bge-reranker-v2-m3"
CROSSENC_QWEN = "Qwen/Qwen3-Reranker-4B"
CROSSENC_VOYAGE = "rerank-2.5"
CROSSENC_COHERE = "rerank-v4.0-pro"

# Experiment Settings
RERANK_TOP_K = 100           # Use 250 for Experiment 2
RRF_K = 60
WANDB_PROJECT = "talentclef-task-b"

# Paths
DATA_DIR = "taskB"
OUTPUT_DIR = "outputs"
DESC_DIR = "descriptions"
