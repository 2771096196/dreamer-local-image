"""Protocol/queue checks with an explicit fake engine; no model or paid calls."""
import asyncio
import base64
import io
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from service import Cancelled, ImageRequest, TaskStore, QwenEngine, create_app


class FakeEngine:
    def __init__(self, slow=False):
        self.calls = []
        self.slow = slow
        self.started = threading.Event()

    def load(self):
        pass

    def unload(self):
        pass

    def status(self):
        return {"state": "ready", "offload": "test"}

    def generate(self, request, directory, progress, cancelled):
        self.calls.append(request)
        self.started.set()
        if self.slow:
            for _ in range(100):
                if cancelled():
                    raise Cancelled()
                time.sleep(.01)
        image = Image.new("RGBA", (request.width, request.height), (72, 84, 100, 128))
        image.save(directory / "1.png")
        progress(1, request.steps)
        return [{"file": "1.png", "seed": request.seed, "width": image.width, "height": image.height, "mode": "RGBA"}]


def wait_task(client, task_id):
    for _ in range(200):
        value = client.get('/v1/tasks/' + task_id).json()
        if value['status'] in {'succeeded', 'failed', 'cancelled'}:
            return value
        time.sleep(.01)
    raise AssertionError('worker did not finish')


def test_task_idempotency_download_restart_and_origins(tmp_path):
    store = TaskStore(tmp_path, FakeEngine())
    with TestClient(create_app(store)) as client:
        assert client.get('/health').json()['protocol'] == 'dreamer-local-image-v1'
        assert client.get('/v1/capabilities').json()['max_input_images'] == 10
        body = {'prompt': 'test', 'request_id': 'once', 'width': 256, 'height': 256}
        task = client.post('/v1/images/generations', json=body).json()
        assert client.post('/v1/images/generations', json=body).json()['id'] == task['id']
        assert client.post('/v1/images/generations', json=dict(body, prompt='changed')).status_code == 409
        result = wait_task(client, task['id'])
        assert result['status'] == 'succeeded'
        response = client.get(result['images'][0]['url'])
        assert Image.open(io.BytesIO(response.content)).mode == 'RGBA'
        assert client.post('/v1/model/load', headers={'Origin': 'https://evil.example'}).status_code == 403
        assert client.get('/health', headers={'Host': 'evil.example'}).status_code == 400
        assert client.post('/v1/images/edits', json={'prompt': 'x', 'images': ['http://127.0.0.1/secret']}).status_code == 422
    assert len(store.engine.calls) == 1
    reopened = TaskStore(tmp_path, FakeEngine())
    assert reopened.get(task['id'])['status'] == 'succeeded'


def test_queued_and_running_cancellation(tmp_path):
    engine = FakeEngine(slow=True)
    with TestClient(create_app(TaskStore(tmp_path, engine))) as client:
        first = client.post('/v1/images/generations', json={'prompt': 'a'}).json()
        assert engine.started.wait(3)
        second = client.post('/v1/images/generations', json={'prompt': 'b'}).json()
        assert client.delete('/v1/tasks/' + second['id']).json()['status'] == 'cancelled'
        client.delete('/v1/tasks/' + first['id'])
        assert wait_task(client, first['id'])['status'] == 'cancelled'
    assert len(engine.calls) == 1


def test_interrupted_jobs_are_not_resubmitted(tmp_path):
    store = TaskStore(tmp_path, FakeEngine())
    task = store.submit(ImageRequest(prompt='never started', request_id='restart'))
    restored = TaskStore(tmp_path, FakeEngine())
    assert restored.get(task['id'])['status'] == 'failed'
    assert restored.submit(ImageRequest(prompt='never started', request_id='restart'))['id'] == task['id']
    assert restored.pending.empty()






def test_mask_composite_preserves_unedited_pixels(tmp_path, monkeypatch):
    def encoded(image):
        buffer = io.BytesIO()
        image.save(buffer, 'PNG')
        return base64.b64encode(buffer.getvalue()).decode()
    original = Image.new('RGBA', (256, 256), (10, 20, 30, 64))
    mask = Image.new('L', original.size, 0)
    mask.paste(255, (0, 0, 128, 256))
    calls = []
    def pipe(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(images=[Image.new('RGBA', original.size, (200, 100, 50, 255))])
    engine = QwenEngine()
    engine.pipe = pipe
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        Generator=lambda *_: SimpleNamespace(manual_seed=lambda seed: seed)))
    result = engine.generate(ImageRequest(prompt='edit', images=[encoded(original)], mask=encoded(mask),
        width=256, height=256, seed=12), tmp_path, lambda *args: None, lambda: False)
    with Image.open(tmp_path / result[0]['file']) as output:
        assert output.getpixel((32, 32)) == (200, 100, 50, 255)
        assert output.getpixel((200, 32)) == (10, 20, 30, 64)
    assert len(calls[0]['image']) == 2
    assert 'edit mask' in calls[0]['prompt']
