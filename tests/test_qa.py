import types

import pytest

import src.config as cfg
from src.qa.answer import SYSTEM, answer, build_messages, parse_refs
from src.qa.llm import LLM, Pacer


class HttpError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        msg = types.SimpleNamespace(content=reply)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(cfg.settings, "cache_dir", tmp_path / "cache")
    monkeypatch.setattr("src.qa.llm.time.sleep", lambda s: None)


MSGS = [{"role": "user", "content": "hi"}]
ONE = [("gemini", "m1")]
TWO = [("gemini", "m1"), ("groq", "m2")]


def test_cache_avoids_second_call():
    client = FakeClient(["hello"])
    llm = LLM(ONE, {"gemini": client})
    assert llm.complete(MSGS) == "hello"
    assert llm.complete(MSGS) == "hello"
    assert client.calls == 1


def test_retries_on_rate_limit():
    client = FakeClient([HttpError(429), HttpError(503), "ok"])
    assert LLM(ONE, {"gemini": client}).complete(MSGS) == "ok"
    assert client.calls == 3


def test_non_retryable_error_raises():
    client = FakeClient([HttpError(400)])
    with pytest.raises(HttpError):
        LLM(ONE, {"gemini": client}).complete(MSGS)


def test_falls_back_then_skips_cooled_down_provider(monkeypatch):
    monkeypatch.setattr(cfg.settings, "llm_max_retries", 2)
    primary = FakeClient([HttpError(429), HttpError(429)])
    backup = FakeClient(["from backup", "again"])
    llm = LLM(TWO, {"gemini": primary, "groq": backup})
    assert llm.complete(MSGS) == "from backup"
    assert llm.last_ref == ("groq", "m2")
    other = [{"role": "user", "content": "second"}]
    assert llm.complete(other) == "again"
    assert primary.calls == 2


def test_whole_chain_is_retried_after_a_cooldown(monkeypatch):
    monkeypatch.setattr(cfg.settings, "llm_max_retries", 1)
    monkeypatch.setattr(cfg.settings, "llm_cooldown_s", 0.0)
    primary = FakeClient([HttpError(429), "second round"])
    llm = LLM(ONE, {"gemini": primary})
    assert llm.complete(MSGS) == "second round" and primary.calls == 2
    monkeypatch.setattr(cfg.settings, "llm_chain_rounds", 2)
    always = FakeClient([HttpError(429), HttpError(429)])
    with pytest.raises(RuntimeError, match="All providers failed"):
        LLM(ONE, {"gemini": always}).complete([{"role": "user", "content": "other"}])


def test_payment_required_everywhere_fails_without_waiting(monkeypatch):
    monkeypatch.setattr(cfg.settings, "llm_cooldown_s", 1000.0)
    only = FakeClient([HttpError(402)])
    with pytest.raises(RuntimeError, match="All providers failed"):
        LLM(ONE, {"gemini": only}).complete(MSGS)
    assert only.calls == 1


def test_timeouts_count_as_retryable():
    from src.qa.llm import retryable

    class APIConnectionError(Exception):
        pass

    class APITimeoutError(APIConnectionError):
        pass

    assert retryable(APITimeoutError()) and retryable(APIConnectionError()) and not retryable(ValueError())


def test_payment_required_falls_back_without_retrying(monkeypatch):
    monkeypatch.setattr(cfg.settings, "llm_max_retries", 3)
    primary = FakeClient([HttpError(402)])
    backup = FakeClient(["from backup", "again"])
    llm = LLM(TWO, {"gemini": primary, "groq": backup})
    assert llm.complete(MSGS) == "from backup" and primary.calls == 1
    assert llm.complete([{"role": "user", "content": "second"}]) == "again" and primary.calls == 1


def test_cached_answer_from_any_chain_member_is_reused():
    backup = FakeClient(["cached by backup"])
    LLM([("groq", "m2")], {"groq": backup}).complete(MSGS)
    primary = FakeClient([])
    llm = LLM(TWO, {"gemini": primary, "groq": backup})
    assert llm.complete(MSGS) == "cached by backup"
    assert primary.calls == 0 and backup.calls == 1


class Clock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_pacer_spaces_requests_and_limits_tokens(monkeypatch):
    clock = Clock()
    monkeypatch.setattr("src.qa.llm.time", types.SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep))
    pacer = Pacer(rpm=5, tpm=1000)
    pacer.wait(100)
    pacer.wait(100)
    assert clock.t == pytest.approx(12.0)
    pacer = Pacer(rpm=None, tpm=1000)
    start = clock.t
    pacer.wait(600)
    pacer.wait(600)
    assert clock.t - start == pytest.approx(60.0)


def test_parse_refs():
    assert parse_refs("A [1] and B [2, 3]. Again [1].") == [1, 2, 3]
    assert parse_refs("no citations") == []
    assert parse_refs("Uses YOLO【2†L1-L4】 on a Pi【3†L1-L3】") == [2, 3]


def test_answer_maps_citations_and_flags_invalid(index):
    client = FakeClient(["It uses YOLO [1] on camera frames [2]. Also [9]."])
    ans = answer("yolo object detection camera", index, LLM(ONE, {"gemini": client}), k=2, mode="dense")
    assert [c.n for c in ans.citations] == [1, 2]
    assert ans.citations[0].doc_id == "A" and ans.citations[0].chunk_id == "A:0"
    assert ans.invalid_refs == [9]
    assert "p. 1" in str(ans.citations[0])


def test_refusal_instruction_only_when_enabled(index):
    hits = index.search("budget", k=1, mode="bm25")
    plain = build_messages("q", hits)[0]["content"]
    refuse = build_messages("q", hits, allow_refusal=True)[0]["content"]
    assert plain == SYSTEM and "cannot find" in refuse


def test_bare_prompt_has_no_grounding_or_refusal_instruction(index):
    hits = index.search("budget", k=1, mode="bm25")
    bare = build_messages("q", hits, prompt="bare")[0]["content"]
    assert "outside knowledge" not in bare and "cannot find" not in bare and "[2]" in bare
    assert "outside knowledge" in build_messages("q", hits)[0]["content"]
