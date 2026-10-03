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

    def __init__(self):
        self.calls = 0

    def create_voice_clone_prompt(self, ref_audio, ref_text=None, x_vector_only_mode=False):
        return [VoiceClonePromptItem(ref_code=torch.zeros(10, 16, dtype=torch.long), ref_spk_embedding=torch.zeros(8),
                                     x_vector_only_mode=x_vector_only_mode, icl_mode=not x_vector_only_mode,
                                     ref_text=ref_text)]

    def generate_voice_clone(self, text, language, voice_clone_prompt, non_streaming_mode, max_new_tokens, **kw):
        self.calls += 1
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


def _make_voice(client, name="narrator"):
    buf = io.BytesIO()
    t = np.arange(SR * 6) / SR
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
