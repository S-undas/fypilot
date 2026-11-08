from __future__ import annotations

import os

from src.abstention.claims import split_claims
from src.config import settings

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


class NLIScorer:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.nli_model
        self._model = None
        self._entail = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device="cpu")
            labels = {int(k): v.lower() for k, v in self._model.config.id2label.items()}
            self._entail = next(i for i, name in labels.items() if "entail" in name)
        return self._model

    def entailment(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Probability that each premise entails its hypothesis. Pairs are (premise, hypothesis)."""
        if not pairs:
            return []
        probs = self.model.predict(pairs, apply_softmax=True, batch_size=16, show_progress_bar=False)
        return [float(p[self._entail]) for p in probs]


SUPPORTED = 0.5


WINDOW_WORDS = 55
STRIDE_WORDS = 28


def windows(text: str, size: int = WINDOW_WORDS, stride: int = STRIDE_WORDS) -> list[str]:
    """Overlapping word windows. NLI models are trained on short pairs, so a fact inside a long passage
    is checked against the short stretch of text around it."""
    words = text.split()
    if len(words) <= size:
        return [text]
    starts = list(range(0, len(words) - size + stride, stride))
    return [" ".join(words[s:s + size]) for s in starts]


def claim_probs(scorer, answer: str, premises: list[str], window: bool = True) -> list[float] | None:
    """Best entailment probability over the premise windows, for each factual sentence of the answer.
    None when the answer has no factual sentences (a plain refusal)."""
    claims = split_claims(answer)
    if window:
        premises = [w for p in premises for w in windows(p)]
    if not claims or not premises:
        return None
    probs = scorer.entailment([(p, c) for c in claims for p in premises])
    n = len(premises)
    return [max(probs[i * n:(i + 1) * n]) for i in range(len(claims))]


def support_score(scorer, answer: str, premises: list[str]) -> float | None:
    """Weakest sentence: every sentence must be entailed."""
    probs = claim_probs(scorer, answer, premises)
    return None if probs is None else min(probs)


def support_scores(scorer, answer: str, premises: list[str]) -> dict | None:
    """Both aggregates: weakest sentence ('min') and the share of sentences entailed ('frac')."""
    probs = claim_probs(scorer, answer, premises)
    if probs is None:
        return None
    return {"min": min(probs), "frac": sum(p >= SUPPORTED for p in probs) / len(probs), "claims": len(probs)}
