import pytest

from src.config import PROVIDERS, Settings


def test_default_models():
    s = Settings()
    assert s.embedding_model == "BAAI/bge-small-en-v1.5"
    assert s.nli_model == "cross-encoder/nli-deberta-v3-base"


def test_thresholds_start_unset():
    s = Settings()
    assert s.score_threshold is None
    assert s.nli_entailment_threshold is None


def test_providers_registered_without_keys():
    s = Settings()
    for provider, _ in [*s.llm_chain, s.judge_model]:
        assert provider in PROVIDERS
    for p in PROVIDERS.values():
        assert p.api_key_env.endswith("_API_KEY")
        assert not hasattr(p, "api_key")


def test_missing_key_raises(monkeypatch, tmp_path):
    import src.config as cfg

    monkeypatch.setattr(cfg, "ROOT", tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        Settings().api_key("gemini")
