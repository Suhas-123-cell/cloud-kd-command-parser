"""Unit tests for sharding, cleaning, schema validation and the service API (no model, no cluster)."""
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "worker"), str(ROOT / "service")]

from clean import classify, clean  # noqa: E402
from inference import prompt_for, training_text, validate  # noqa: E402
from worker import select_shard  # noqa: E402


def rec(i, text, raw):
    return {"id": f"C{i:04d}", "text": text, "teacher_raw": raw}


GOOD = '{"intent": "weather_query", "date": null, "time": null, "place": "london"}'


# ---------- sharding ----------
@pytest.mark.parametrize("count", [1, 2, 3, 7])
def test_shards_cover_every_line_exactly_once(count):
    lines = [json.dumps({"id": f"C{i:04d}", "text": f"t{i}"}) for i in range(100)]
    seen = [r["id"] for k in range(count) for r in select_shard(lines, k, count)]
    assert sorted(seen) == sorted(f"C{i:04d}" for i in range(100))
    assert len(seen) == len(set(seen))


def test_shard_uses_line_number_modulo():
    lines = [json.dumps({"id": f"X{i}", "text": ""}) for i in range(10)]
    assert [r["id"] for r in select_shard(lines, 1, 3)] == ["X1", "X4", "X7"]
    with pytest.raises(ValueError):
        select_shard(lines, 3, 3)


# ---------- cleaning ----------
def test_classify_each_rejection_reason():
    assert classify(rec(1, "what is the weather in london", GOOD)) is None
    assert classify(rec(2, "x", '{"intent": "weather_query", "date": nul')) == "a"
    assert classify(rec(3, "x", '{"intent": "weather_query", "date": null, "time": null}')) == "b"
    assert classify(rec(4, "x", '["intent", "date", "time", "place"]')) == "b"
    assert classify(rec(5, "x", '{"intent": "null", "date": null, "time": null, "place": null}')) == "c"
    invented = '{"intent": "alarm_query", "date": "today", "time": null, "place": null}'
    assert classify(rec(6, "do i have any alarms", invented)) == "d"
    numeric = '{"intent": "alarm_set", "date": null, "time": 7, "place": null}'
    assert classify(rec(7, "wake me at 7", numeric)) == "d"


def test_slot_match_ignores_case_and_spacing():
    raw = '{"intent": "weather_query", "date": null, "time": null, "place": "New  York"}'
    assert classify(rec(1, "weather in new york please", raw)) is None


def test_clean_dedup_keeps_smallest_id_and_counts_add_up():
    records = [
        rec(3, "what is the weather in london", GOOD),
        rec(1, "What is the weather in  London", GOOD),   # same command, smaller id -> kept
        rec(2, "bad", "not json"),
    ]
    kept, counts = clean(records)
    assert [r["id"] for r in kept] == ["C0001"]
    assert counts == {"input": 3, "a_invalid_json": 1, "b_wrong_keys": 0, "c_bad_intent": 0,
                      "d_bad_slot": 0, "duplicates": 1, "kept": 1}


def test_clean_output_is_in_id_order_and_raw_untouched():
    raw = GOOD + "\n"
    kept, _ = clean([rec(10, "weather in london", raw), rec(2, "london weather", raw)])
    assert [r["id"] for r in kept] == ["C0002", "C0010"]
    assert kept[0]["teacher_raw"] == raw


# ---------- schema validation / prompt format ----------
def test_validate_accepts_only_the_four_key_schema():
    assert validate(GOOD) == json.loads(GOOD)
    assert validate("not json") is None
    assert validate('{"intent": "dance", "date": null, "time": null, "place": null}') is None
    assert validate('{"intent": "weather_query", "date": null, "time": null, "place": null, "x": 1}') is None
    assert validate('{"intent": "weather_query", "date": 5, "time": null, "place": null}') is None


def test_training_text_matches_the_assignment_format():
    prompt, target = training_text("set an alarm", GOOD + "\n")
    assert prompt == "Command: set an alarm\nJSON:"
    assert prompt + target == f"Command: set an alarm\nJSON: {GOOD}"
    assert prompt_for("  hi ") == "Command: hi\nJSON:"


# ---------- service API (fake model) ----------
class FakeParser:
    def __init__(self, out):
        self.out = out

    def generate(self, text):
        return self.out


@pytest.fixture
def make_client(monkeypatch):
    from fastapi.testclient import TestClient

    def _make(fault="0", output=GOOD, loaded=True):
        monkeypatch.setenv("FAULT_INJECT", fault)
        monkeypatch.setenv("MODEL_VERSION", "vtest")
        monkeypatch.setenv("POD_NAME", "pod-a")
        import app as app_module
        app_module = importlib.reload(app_module)
        monkeypatch.setattr(app_module, "load_model", lambda: None)  # no S3 in unit tests
        app_module.state["parser"] = FakeParser(output) if loaded else None
        return TestClient(app_module.app)

    return _make


def test_parse_returns_object_and_headers(make_client):
    with make_client() as c:
        r = c.post("/parse", json={"text": "what is the weather in london"})
    assert r.status_code == 200
    assert r.json() == json.loads(GOOD)
    assert r.headers["X-Model-Version"] == "vtest"
    assert r.headers["X-Served-By"] == "pod-a"


def test_fault_inject_gives_500_with_exact_body(make_client):
    with make_client(fault="1") as c:
        r = c.post("/parse", json={"text": "anything"})
    assert r.status_code == 500
    assert r.json() == {"error": "invalid_model_output"}


def test_invalid_model_output_is_never_200(make_client):
    with make_client(output='{"intent": "alarm_set", "date": "tomor') as c:
        r = c.post("/parse", json={"text": "x"})
    assert r.status_code == 500 and r.json() == {"error": "invalid_model_output"}


def test_readyz_503_until_loaded_healthz_always_200(make_client):
    with make_client(loaded=False) as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/readyz").status_code == 503
        assert c.post("/parse", json={"text": "x"}).status_code == 503
    with make_client() as c:
        assert c.get("/readyz").status_code == 200
