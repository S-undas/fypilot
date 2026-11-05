from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent


class LLMProvider(BaseModel):
    base_url: str
    api_key_env: str
    model: str | None = None


PROVIDERS: dict[str, LLMProvider] = {
    "gemini": LLMProvider(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GEMINI_API_KEY",
    ),
    "groq": LLMProvider(
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
    ),
}


class Settings(BaseModel):
    raw_dir: Path = ROOT / "data" / "raw"
    processed_dir: Path = ROOT / "data" / "processed"
    index_dir: Path = ROOT / "index"
    cache_dir: Path = ROOT / ".cache"
    eval_dir: Path = ROOT / "eval"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    reranker_model: str | None = None

    llm_provider: str = "gemini"
    llm_fallback_provider: str = "groq"
    llm_temperature: float = 0.0

    top_k: int = 5

    # None until tuned on a dev split
    score_threshold: float | None = None
    nli_entailment_threshold: float | None = None

    def provider(self, name: str | None = None) -> LLMProvider:
        return PROVIDERS[name or self.llm_provider]

    def api_key(self, name: str | None = None) -> str:
        load_dotenv(ROOT / ".env")
        env = self.provider(name).api_key_env
        key = os.environ.get(env)
        if not key:
            raise RuntimeError(f"{env} is not set. Add it to .env (see .env.example).")
        return key


settings = Settings()
