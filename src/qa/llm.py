from __future__ import annotations

import hashlib
import json
import time
from collections import deque

from src.config import LIMITS, PROVIDERS, settings

RETRY_STATUS = {429, 500, 502, 503, 504}
UNAVAILABLE_STATUS = {402}


def unavailable(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) in UNAVAILABLE_STATUS


def retryable(exc: Exception) -> bool:
    names = {c.__name__ for c in type(exc).__mro__}
    return getattr(exc, "status_code", None) in RETRY_STATUS or "APIConnectionError" in names


def describe(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    if status == 402:
        return "payment required, the free credit is used up"
    if status == 429:
        return "rate limit or daily limit reached"
    if status in RETRY_STATUS:
        return f"the service had an error ({status})"
    if "APIConnectionError" in {c.__name__ for c in type(exc).__mro__}:
        return "could not connect, offline or blocked by the network"
    return type(exc).__name__


class ProvidersUnavailable(RuntimeError):
    def __init__(self, message: str, reasons: dict[str, str]):
        super().__init__(message)
        self.reasons = reasons


class Pacer:
    def __init__(self, rpm: int | None, tpm: int | None):
        self.interval = 60 / rpm if rpm else 0.0
        self.tpm = tpm
        self.next_ok = 0.0
        self.window: deque[tuple[float, int]] = deque()

    def wait(self, tokens: int) -> None:
        now = time.monotonic()
        if now < self.next_ok:
            time.sleep(self.next_ok - now)
            now = time.monotonic()
        if self.tpm:
            while True:
                while self.window and now - self.window[0][0] >= 60:
                    self.window.popleft()
                if not self.window or sum(n for _, n in self.window) + tokens <= self.tpm:
                    break
                time.sleep(60 - (now - self.window[0][0]))
                now = time.monotonic()
        self.next_ok = now + self.interval
        self.window.append((now, tokens))


class LLM:
    def __init__(self, chain: list[tuple[str, str]] | None = None, clients: dict | None = None):
        self.chain = chain or settings.llm_chain
        self.clients = clients or {}
        self.pacers: dict[tuple[str, str], Pacer] = {}
        self.cooldown: dict[str, float] = {}
        self.calls = 0
        self.last_ref: tuple[str, str] | None = None

    def client(self, provider: str):
        if provider not in self.clients:
            from openai import OpenAI

            self.clients[provider] = OpenAI(
                base_url=PROVIDERS[provider].base_url, api_key=settings.api_key(provider), max_retries=0
            )
        return self.clients[provider]

    def pacer(self, ref: tuple[str, str]) -> Pacer:
        if ref not in self.pacers:
            rpm, tpm = LIMITS.get(ref, (None, None))
            self.pacers[ref] = Pacer(rpm, tpm)
        return self.pacers[ref]

    @staticmethod
    def cache_path(ref: tuple[str, str], messages: list[dict], temperature: float):
        key = json.dumps([*ref, messages, temperature], sort_keys=True)
        return settings.cache_dir / "llm" / f"{hashlib.sha256(key.encode()).hexdigest()}.json"

    def call(self, ref: tuple[str, str], messages: list[dict], temperature: float, max_tokens: int | None) -> str:
        provider, model = ref
        est = len(json.dumps(messages)) // 4 + (max_tokens or settings.llm_out_tokens_estimate)
        for attempt in range(settings.llm_max_retries):
            self.pacer(ref).wait(est)
            try:
                resp = self.client(provider).chat.completions.create(
                    model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
                )
                self.calls += 1
                return resp.choices[0].message.content or ""
            except Exception as exc:
                if unavailable(exc) or not retryable(exc) or attempt == settings.llm_max_retries - 1:
                    raise
                time.sleep(min(2**attempt, 60))
        raise RuntimeError("unreachable")

    def complete(self, messages: list[dict], temperature: float | None = None, max_tokens: int | None = None) -> str:
        temperature = settings.llm_temperature if temperature is None else temperature
        for ref in self.chain:
            path = self.cache_path(ref, messages, temperature)
            if path.exists():
                self.last_ref = ref
                return json.loads(path.read_text(encoding="utf-8"))["text"]
        last_exc: Exception | None = None
        reasons: dict[str, str] = {}
        for round_ in range(settings.llm_chain_rounds):
            now = time.monotonic()
            order = [r for r in self.chain if self.cooldown.get(r[0], 0) <= now] or list(self.chain)
            waitable = False
            for ref in order:
                try:
                    text = self.call(ref, messages, temperature, max_tokens)
                except Exception as exc:
                    if not (retryable(exc) or unavailable(exc)):
                        raise
                    waitable = waitable or not unavailable(exc)
                    self.cooldown[ref[0]] = time.monotonic() + (3600.0 if unavailable(exc) else settings.llm_cooldown_s)
                    last_exc = exc
                    reasons[ref[0]] = describe(exc)
                    continue
                path = self.cache_path(ref, messages, temperature)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"text": text}), encoding="utf-8")
                self.last_ref = ref
                return text
            if not waitable or round_ == settings.llm_chain_rounds - 1:
                break
            time.sleep(settings.llm_cooldown_s)
        raise ProvidersUnavailable(f"All providers failed: {last_exc}", reasons)
