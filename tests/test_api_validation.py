import base64
import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from service import ImageRequest, TaskStore, create_app
from test_service import FakeEngine, wait_task


@pytest.mark.parametrize("change", [
    {"width": 257}, {"height": 128}, {"width": 4096, "height": 4096},
    {"width": 1360}, {"height": 272},
    {"n": 0}, {"n": 9}, {"steps": 0}, {"steps": 101}, {"seed": -2},
    {"negative_prompt": "noise", "guidance_scale": 1}, {"model": "another-model"},
    {"background": "invalid"}, {"unknown_option": True}, {"mask": "not-an-image"},
    {"images": ["https://example.com/image.png"]},
])
def test_bad_input_is_rejected_before_queue(tmp_path, change):
    engine = FakeEngine()
    store = TaskStore(tmp_path, engine)
    with TestClient(create_app(store)) as client:
        response = client.post("/v1/images/generations", json={"prompt": "test", **change})
        assert response.status_code == 422
        assert client.get("/v1/tasks").json()["data"] == []
    assert not engine.calls


def test_input_capacity_and_png_alpha():
    buffer = io.BytesIO()
    Image.new("RGBA", (16, 16), (1, 2, 3, 64)).save(buffer, "PNG")
    image = base64.b64encode(buffer.getvalue()).decode()
    assert len(ImageRequest(prompt="test", images=[image] * 10).images) == 10
    with pytest.raises(ValueError):
        ImageRequest(prompt="test", images=[image] * 11)
    with pytest.raises(ValueError):
        ImageRequest(prompt="test", images=[image] * 10, mask=image)
    with pytest.raises(ValueError):
        ImageRequest(prompt="test", images=[image], mask=image, background="transparent")


def test_engine_failure_keeps_readable_receipt(tmp_path):
    class FailingEngine(FakeEngine):
        def load(self):
            raise RuntimeError("CUDA out of memory")
    with TestClient(create_app(TaskStore(tmp_path, FailingEngine()))) as client:
        task = client.post('/v1/images/generations', json={'prompt': 'test'}).json()
        receipt = wait_task(client, task['id'])
        assert receipt['status'] == 'failed'
        assert receipt['error'] == 'CUDA out of memory'
        assert receipt['images'] == []
        assert client.get(f"/v1/tasks/{task['id']}/images/1").status_code == 404


def test_model_management_and_api_documentation(tmp_path):
    with TestClient(create_app(TaskStore(tmp_path, FakeEngine()))) as client:
        for action in ['load', 'unload']:
            task = client.post('/v1/model/' + action).json()
            assert wait_task(client, task['id'])['status'] == 'succeeded'
        assert client.get('/').status_code == 200
        assert client.get('/docs').status_code == 200
        assert client.get('/openapi.json').json()['paths']['/v1/images/edits']
        assert client.get('/v1/models').json()['data'][0]['id'] == 'Qwen/Qwen-Image-2.1'
        assert client.get('/v1/tasks/missing').status_code == 404
