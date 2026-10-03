import io
import time
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
from server.app import create_app
from server.db import Store
from server.engine import Engine
from server.worker import Worker

SR = 24000


class FakeTTS:
    """Stands in for Qwen3TTSModel: audio length tracks text length like real speech."""

    device = "cpu"
    model = SimpleNamespace(
        config=SimpleNamespace(_name_or_path="fake"), tts_model_type="base", tts_model_size="1b7", dtype="fp32",
    )

    def __init__(self, delay: float = 0.0):
        self.calls = 0
        self.delay = delay

    def create_voice_clone_prompt(self, ref_audio, ref_text=None, x_vector_only_mode=False):
        return [VoiceClonePromptItem(ref_code=torch.zeros(10, 16, dtype=torch.long), ref_spk_embedding=torch.zeros(8),
                                     x_vector_only_mode=x_vector_only_mode, icl_mode=not x_vector_only_mode,
                                     ref_text=ref_text)]

    def generate_voice_clone(self, text, language, voice_clone_prompt, non_streaming_mode, max_new_tokens, **kw):
        self.calls += 1
        time.sleep(self.delay)
        out = []
        for t in text:
            n = int(SR * len(t) / 12.0)
            out.append((0.2 * np.sin(2 * np.pi * 200 * np.arange(n) / SR)).astype(np.float32))
        return out, SR


def _wait(client, job_id, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = client.get(f"/v1/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "partial", "failed"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} stuck: {job}")


@pytest.fixture()
def client(tmp_path):
    fake = FakeTTS()
    store = Store(str(tmp_path / "t.db"))
    engine = Engine(lambda: fake)
    worker = Worker(store, engine, tmp_path)
    app = create_app(store, engine, worker, tmp_path)
    worker.start()
    with TestClient(app) as c:
        c.fake = fake
        yield c
    worker.shutdown()


def _make_voice(client, name="narrator", seconds=6):
    buf = io.BytesIO()
    t = np.arange(int(SR * seconds)) / SR
    sf.write(buf, (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), SR, format="WAV")
    r = client.post("/v1/voices", data={"name": name, "ref_text": "A calm reference sentence for testing the voice."},
                    files={"file": ("ref.wav", buf.getvalue(), "audio/wav")})
    assert r.status_code == 202, r.text
    job = _wait(client, r.json()["id"])
    assert job["status"] == "succeeded", job
    return job


def test_voice_create_then_tts_then_regenerate(client):
    _make_voice(client)
    voices = client.get("/v1/voices").json()
    assert voices[0]["name"] == "narrator" and voices[0]["latest"] == 1

    text = "First paragraph sentence one. Sentence two.\n\nSecond paragraph here."
    r = client.post("/v1/tts/jobs", json={"voice": "narrator", "text": text})
    assert r.status_code == 202
    job = _wait(client, r.json()["id"])
    assert job["status"] == "succeeded", job
    assert job["request"]["voice"] == "narrator/v1"  # pinned at submit time
    assert [c["boundary"] for c in job["result"]["chunks"]] == ["paragraph", "end"]

    audio = client.get(f"/v1/tts/jobs/{job['id']}/audio")
    assert audio.status_code == 200
    wav, sr = sf.read(io.BytesIO(audio.content))
    assert sr == SR and abs(len(wav) / sr - job["result"]["meta"]["duration_s"]) < 0.01

    r = client.post(f"/v1/tts/jobs/{job['id']}/chunks/1/regenerate", json={"seed": 7})
    assert r.status_code == 202
    job2 = _wait(client, job["id"])
    assert job2["result"]["audio_version"] == 2
    assert job2["result"]["chunks"][1]["seed"] == 7
    assert job2["result"]["regenerations"][0]["idx"] == 1


def test_idempotency_key_returns_same_job(client):
    _make_voice(client)
    body = {"voice": "narrator", "text": "Hello there."}
    a = client.post("/v1/tts/jobs", json=body, headers={"Idempotency-Key": "abc"}).json()
    b = client.post("/v1/tts/jobs", json=body, headers={"Idempotency-Key": "abc"}).json()
    assert a["id"] == b["id"]


def test_unknown_voice_and_bad_input(client):
    assert client.post("/v1/tts/jobs", json={"voice": "nope", "text": "hi"}).status_code == 404
    _make_voice(client)
    assert client.post("/v1/tts/jobs", json={"voice": "narrator", "text": "  "}).status_code == 422
    assert client.post("/v1/voices", data={"name": "Bad Name", "ref_text": "x"},
                       files={"file": ("r.wav", b"x", "audio/wav")}).status_code == 422


def test_preview_chunks_does_not_need_model(client):
    r = client.post("/v1/tts/preview-chunks", json={"text": "One. Two.\n\nThree.", "lexicon": {"Two": "Too"}})
    body = r.json()
    assert body["text_normalized"].startswith("One. Too.")
    assert [c["boundary"] for c in body["chunks"]] == ["paragraph", "end"]
    assert client.get("/v1/engine").json()["state"] == "stopped"


def test_engine_start_stop(client):
    client.post("/v1/engine/start")
    for _ in range(100):
        if client.get("/v1/engine").json()["state"] == "ready":
            break
        time.sleep(0.02)
    assert client.get("/v1/engine").json()["state"] == "ready"
    assert client.post("/v1/engine/stop").json()["state"] == "stopped"


def test_api_key_enforced(tmp_path):
    store = Store(str(tmp_path / "k.db"))
    engine = Engine(lambda: FakeTTS())
    app = create_app(store, engine, Worker(store, engine, tmp_path), tmp_path, api_key="s3cret")
    with TestClient(app) as c:
        assert c.get("/v1/engine").status_code == 401
        assert c.get("/v1/engine", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_store_is_safe_under_concurrent_reads(tmp_path):
    # Regression: rows were fetched after the lock was released, so a worker
    # thread polling queue_depth() could clobber another thread's result.
    import threading

    store = Store(str(tmp_path / "c.db"))
    job = store.create_job("tts", {"text": "x"}, "tts.render", {})
    errors = []

    def hammer():
        try:
            for _ in range(500):
                assert store.queue_depth() >= 0
                assert store.get_job(job["id"])["id"] == job["id"]
                store.list_jobs("tts", limit=5)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=hammer) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[:3]


def _upload(client, name, payload, ref_text="Some words here."):
    r = client.post("/v1/voices", data={"name": name, "ref_text": ref_text},
                    files={"file": ("ref.wav", payload, "audio/wav")})
    return r


def test_bad_uploads_fail_with_clear_errors(client, tmp_path):
    job = _wait(client, _upload(client, "garbage", b"definitely not audio").json()["id"])
    assert job["status"] == "failed" and job["error"]

    buf = io.BytesIO()
    sf.write(buf, (0.2 * np.sin(np.arange(SR * 35) / 8)).astype(np.float32), SR, format="WAV")
    job = _wait(client, _upload(client, "toolong", buf.getvalue()).json()["id"])
    assert job["status"] == "failed" and "at most 30s" in job["error"]

    buf = io.BytesIO()
    sf.write(buf, (0.2 * np.sin(np.arange(SR * 1) / 8)).astype(np.float32), SR, format="WAV")
    job = _wait(client, _upload(client, "tooshort", buf.getvalue()).json()["id"])
    assert job["status"] == "failed" and "at least 3s" in job["error"]

    # Raw uploads are removed once processed, success or failure.
    assert list((tmp_path / "uploads").iterdir()) == []


def test_long_reference_warns(client):
    job = _make_voice(client, "longref", seconds=24)
    assert job["status"] == "succeeded"
    v = client.get("/v1/voices").json()[0]["versions"][0]
    assert v["warnings"] and "24s" in v["warnings"][0]


def test_upload_size_limit(client, monkeypatch):
    import server.app as app_mod

    monkeypatch.setattr(app_mod, "MAX_UPLOAD_BYTES", 1000)
    r = _upload(client, "huge", b"x" * 5000)
    assert r.status_code == 413


@pytest.fixture()
def slow_client(tmp_path):
    fake = FakeTTS(delay=0.6)
    store = Store(str(tmp_path / "s.db"))
    engine = Engine(lambda: fake)
    worker = Worker(store, engine, tmp_path)
    app = create_app(store, engine, worker, tmp_path)
    worker.start()
    with TestClient(app) as c:
        c.engine = engine
        yield c
    worker.shutdown()


def test_regenerate_while_running_is_rejected(slow_client):
    _make_voice(slow_client)
    jid = slow_client.post("/v1/tts/jobs", json={"voice": "narrator", "text": "One. Two.\n\nThree. Four."}).json()["id"]
    for _ in range(100):
        if slow_client.get(f"/v1/jobs/{jid}").json()["status"] == "running":
            break
        time.sleep(0.02)
    r = slow_client.post(f"/v1/tts/jobs/{jid}/chunks/0/regenerate", json={})
    assert r.status_code == 409
    assert _wait(slow_client, jid)["status"] == "succeeded"


def test_stop_while_busy_finishes_job_then_unloads(slow_client):
    _make_voice(slow_client)
    jid = slow_client.post("/v1/tts/jobs", json={"voice": "narrator", "text": "One sentence. Another sentence."}).json()["id"]
    for _ in range(200):
        if slow_client.get("/v1/engine").json()["state"] == "busy":
            break
        time.sleep(0.01)
    st = slow_client.post("/v1/engine/stop").json()
    assert st["state"] == "busy" and st["stop_pending"]
    assert _wait(slow_client, jid)["status"] == "succeeded"
    time.sleep(0.3)
    assert slow_client.get("/v1/engine").json()["state"] == "stopped"


def test_stop_while_loading_unloads_after_load(tmp_path):
    def slow_loader():
        time.sleep(0.3)
        return FakeTTS()

    engine = Engine(slow_loader)
    engine.start()
    assert engine.state == "loading"
    engine.stop()
    time.sleep(0.6)
    assert engine.state == "stopped" and engine.tts is None


def test_crashed_tasks_are_requeued_on_restart(tmp_path):
    db = str(tmp_path / "r.db")
    store = Store(db)
    job = store.create_job("tts", {"text": "x"}, "tts.render", {})
    task = store.claim_task()
    store.update_job(job["id"], status="running")
    assert task and store.claim_task() is None  # claimed, simulating a crash mid-render
    restarted = Store(db)
    assert restarted.get_job(job["id"])["status"] == "queued"
    assert restarted.claim_task()["job_id"] == job["id"]


def test_job_list_is_slim_and_hides_tagged_jobs(client):
    _make_voice(client)
    mine = client.post("/v1/tts/jobs", json={"voice": "narrator", "text": "A normal job. " * 40}).json()["id"]
    bench = client.post("/v1/tts/jobs", json={"voice": "narrator", "text": "A bench job.", "tag": "bench"}).json()["id"]
    _wait(client, mine), _wait(client, bench)

    default = client.get("/v1/tts/jobs").json()
    assert [j["id"] for j in default] == [mine]
    assert "chunks" not in default[0]["result"] and default[0]["result"]["meta"]["duration_s"] > 0
    assert len(default[0]["request"]["text"]) <= 300
    assert [j["id"] for j in client.get("/v1/tts/jobs?tag=bench").json()] == [bench]
    assert {j["id"] for j in client.get("/v1/tts/jobs?all=true").json()} == {mine, bench}
