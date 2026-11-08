from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl_cache"))


class LLMProvider(BaseModel):
    base_url: str
    api_key_env: str


PROVIDERS: dict[str, LLMProvider] = {
    "cerebras": LLMProvider(base_url="https://api.cerebras.ai/v1", api_key_env="CEREBRAS_API_KEY"),
    "gemini": LLMProvider(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key_env="GEMINI_API_KEY"
    ),
    "groq": LLMProvider(base_url="https://api.groq.com/openai/v1", api_key_env="GROQ_API_KEY"),
}

# (requests per minute, tokens per minute) from each provider's limits page.
# Gemini values are read off a chart and approximate.
LIMITS: dict[tuple[str, str], tuple[int, int]] = {
    ("cerebras", "gpt-oss-120b"): (5, 30_000),
    ("cerebras", "qwen-3.8-27b"): (450, 150_000),
    ("groq", "openai/gpt-oss-120b"): (30, 8_000),
    ("groq", "qwen/qwen3.8-27b"): (30, 8_000),
    ("gemini", "gemini-3.5-flash-lite"): (15, 250_000),
}


class Settings(BaseModel):
    raw_dir: Path = ROOT / "data" / "raw"
    processed_dir: Path = ROOT / "data" / "processed"
    index_dir: Path = ROOT / "index"
    cache_dir: Path = ROOT / ".cache"
    eval_dir: Path = ROOT / "eval"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    reranker_model: str | None = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    llm_chain: list[tuple[str, str]] = [("cerebras", "qwen-3.8-27b"), ("groq", "qwen/qwen3.8-27b")]
    judge_model: tuple[str, str] = ("gemini", "gemini-3.5-flash-lite")
    llm_temperature: float = 0.0
    llm_max_retries: int = 5
    llm_cooldown_s: float = 60.0
    llm_out_tokens_estimate: int = 800

    chunk_max_chars: int = 1500
    chunk_overlap_chars: int = 150
    chunk_min_chars: int = 40

    top_k: int = 5
    retrieval_mode: str = "hybrid"
    rrf_k: int = 60
    rerank_pool: int = 20

    # None until tuned on a dev split
    score_threshold: float | None = None
    nli_entailment_threshold: float | None = None

    def api_key(self, name: str) -> str:
        load_dotenv(ROOT / ".env")
        env = PROVIDERS[name].api_key_env
        key = os.environ.get(env)
        if not key:
            raise RuntimeError(f"{env} is not set. Add it to .env (see .env.example).")
        return key


settings = Settings()
