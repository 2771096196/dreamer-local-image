"""Standalone, loopback-only Qwen Image worker. Torch loads only in the worker."""
from __future__ import annotations

import base64
import gc
import hashlib
import io
import json
import os
import queue
import secrets
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

MODEL_ID = "Qwen/Qwen-Image-2.1"
ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("DREAMER_LOCAL_IMAGE_DATA", str(ROOT / "data")))
TERMINAL = {"succeeded", "failed", "cancelled"}


class ImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal["Qwen/Qwen-Image-2.1"] = MODEL_ID
    prompt: str = Field(min_length=1, max_length=60000)
    images: list[str] = Field(default_factory=list, max_length=10)
    mask: str | None = None
    width: int = Field(default=1024, ge=256, le=4096, multiple_of=32)
    height: int = Field(default=1024, ge=256, le=4096, multiple_of=32)
    n: int = Field(default=1, ge=1, le=8)
    steps: int = Field(default=40, ge=1, le=100)
    seed: int = Field(default=-1, ge=-1, le=2147483647)
    negative_prompt: str = Field(default="", max_length=12000)
    guidance_scale: float = Field(default=1.0, ge=1.0, le=10.0)
    background: Literal["default", "transparent"] = "default"
    use_kv_cache: bool = True
    request_id: str = Field(default="", max_length=160, pattern=r"^[a-zA-Z0-9_.:-]*$")

    @model_validator(mode="after")
    def validate_combination(self):
        if self.width * self.height > 2048 * 2048:
            raise ValueError("当前服务每张输出最多 4194304 像素，请降低尺寸")
        if self.mask and (not self.images or len(self.images) >= 10):
            raise ValueError("蒙版编辑需要主图，且主图与参考图合计最多 9 张")
        if self.mask and self.background == "transparent":
            raise ValueError("蒙版保留原图与透明背景生成不能同时启用")
        if self.negative_prompt and self.guidance_scale <= 1:
            raise ValueError("负面提示词需要 guidance_scale 大于 1")
        for value in self.images + ([self.mask] if self.mask else []):
            decode_image(value)
        return self


def decode_image(value: str) -> Image.Image:
    """Only accept inline images; never fetch arbitrary URLs or local files."""
    if len(value) > 32 * 1024 * 1024:
        raise ValueError("单张输入图片过大（base64 上限 32 MiB）")
    try:
        raw = base64.b64decode(value.split(",", 1)[-1], validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > 32_000_000:
                raise ValueError("输入图片超过 3200 万像素")
            return ImageOps.exif_transpose(image).convert("RGBA" if "A" in image.getbands() or "transparency" in image.info else "RGB")
    except Exception as error:
        raise ValueError("图片必须是有效的 PNG/JPEG/WebP base64 或 data URL") from error


def capabilities() -> dict:
    return {
        "protocol": "dreamer-local-image-v1", "model": MODEL_ID,
        "text_to_image": True, "image_edit": True, "transparent": True,
        "mask_mode": "reference_and_composite",
        "max_input_images": 10, "max_images_per_task": 8,
        "size_keys": ["1K", "2K"], "max_pixels": 4194304,
        "dimension_multiple": 32, "min_dimension": 256, "max_dimension": 4096,
        "parameters": {"steps": {"min": 1, "max": 100, "default": 40},
                       "seed": {"min": -1, "max": 2147483647, "default": -1},
                       "guidance_scale": {"min": 1, "max": 10, "default": 1},
                       "negative_prompt": True, "use_kv_cache": True},
        "cancel": True, "progress": True,
    }


class Cancelled(Exception):
    pass


class QwenEngine:
    def __init__(self):
        self.pipe = None
        self.state = "unloaded"
        self.error = ""
        self.offload = os.environ.get("DREAMER_LOCAL_IMAGE_OFFLOAD", "model")
        self.quantization = os.environ.get("DREAMER_LOCAL_IMAGE_QUANTIZATION", "nf4")
        self.vae_tile_size = int(os.environ.get("DREAMER_LOCAL_IMAGE_VAE_TILE_SIZE", "1024"))

    def load(self):
        if self.pipe is not None:
            return
        self.state, self.error = "loading", ""
        try:
            if self.quantization == "nf4" and self.offload == "sequential":
                raise ValueError("NF4 请使用 model 或 none 显存模式")
            import torch
            from diffusers import QwenImage21Pipeline
            if not torch.cuda.is_available():
                raise RuntimeError("未检测到可用 NVIDIA CUDA 显卡，请检查驱动和 CUDA 版 PyTorch")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model_path = os.environ.get("DREAMER_LOCAL_IMAGE_MODEL", MODEL_ID)
            if model_path == MODEL_ID:
                from model_download import ensure_model
                model_path = str(ensure_model(ROOT / "models" / "Qwen-Image-2.1"))
            components = {}
            if self.quantization == "nf4":
                from diffusers import QwenImage21Transformer2DModel, BitsAndBytesConfig
                from transformers import Qwen3VLForConditionalGeneration
                from transformers import BitsAndBytesConfig as TextQuantizationConfig
                config = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
                # Quantize components one at a time, then park them on CPU. Both full
                # quantized components need not coexist on the GPU during loading.
                encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_path, subfolder="text_encoder", dtype=dtype,
                    quantization_config=TextQuantizationConfig(**config), device_map={"": "cuda"},
                )
                encoder.to("cpu")
                torch.cuda.empty_cache()
                transformer = QwenImage21Transformer2DModel.from_pretrained(
                    model_path, subfolder="transformer", torch_dtype=dtype,
                    quantization_config=BitsAndBytesConfig(**config), device_map={"": "cuda"},
                )
                transformer.to("cpu")
                torch.cuda.empty_cache()
                components = {"text_encoder": encoder, "transformer": transformer}
            elif self.quantization != "none":
                raise ValueError("量化模式必须是 none 或 nf4")
            pipe = QwenImage21Pipeline.from_pretrained(model_path, torch_dtype=dtype, **components)
            if self.vae_tile_size not in {256, 512, 1024}:
                raise ValueError("VAE 分块大小必须为 256、512 或 1024")
            # Tiny 256px tiles produce visible seams on flat backgrounds. Keep
            # 1K output whole and tile only larger images in the default profile.
            pipe.vae.enable_tiling(
                tile_sample_min_height=self.vae_tile_size,
                tile_sample_min_width=self.vae_tile_size,
                tile_sample_stride_height=self.vae_tile_size * 3 // 4,
                tile_sample_stride_width=self.vae_tile_size * 3 // 4,
            )
            if self.offload == "sequential":
                pipe.enable_sequential_cpu_offload()
            elif self.offload == "model":
                pipe.enable_model_cpu_offload()
            elif self.offload == "none":
                pipe.to("cuda")
            else:
                raise ValueError("显存模式必须是 sequential、model 或 none")
            self.pipe, self.state = pipe, "ready"
        except Exception as error:
            self.state, self.error = "error", str(error)
            raise

    def unload(self):
        self.pipe = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        self.state, self.error = "unloaded", ""

    def status(self) -> dict:
        result = {"state": self.state, "error": self.error, "offload": self.offload,
                  "quantization": self.quantization, "vae_tile_size": self.vae_tile_size}
        # Avoid importing the GPU runtime on the HTTP/event-loop thread.
        import sys
        torch = sys.modules.get("torch")
        if torch and torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            result.update(gpu=torch.cuda.get_device_name(), free_vram_mb=free // 1048576,
                          total_vram_mb=total // 1048576,
                          allocated_vram_mb=torch.cuda.memory_allocated() // 1048576,
                          peak_allocated_vram_mb=torch.cuda.max_memory_allocated() // 1048576,
                          peak_reserved_vram_mb=torch.cuda.max_memory_reserved() // 1048576)
        return result

    def generate(self, request: ImageRequest, output: Path, progress, cancelled) -> list[dict]:
        import torch
        self.load()
        if hasattr(torch, "cuda"):
            torch.cuda.reset_peak_memory_stats()
        images = [decode_image(value) for value in request.images]
        mask = decode_image(request.mask).convert("L") if request.mask else None
        prompt = request.prompt
        if mask is not None:
            if mask.size != images[0].size:
                raise ValueError("蒙版尺寸必须与第一张主图一致；白色编辑，黑色保留")
            images.append(mask.convert("RGB"))
            prompt += "\nThe last reference is an edit mask: edit the white region of the first image only; preserve the black region."
        if request.background == "transparent":
            prompt = "This is an RGBA image with transparency. " + prompt + " The image has alpha channel and the background is transparent."
        results = []
        seed = request.seed if request.seed >= 0 else secrets.randbelow(2147483647)
        for index in range(request.n):
            if cancelled():
                raise Cancelled()
            def on_step(pipe, step, timestep, callback_kwargs):
                if cancelled():
                    raise Cancelled()
                progress((index * request.steps + step + 1) / (request.n * request.steps), step + 1)
                return callback_kwargs
            candidate_seed = (seed + index) % 2147483648
            try:
                image = self.pipe(
                    prompt=prompt, image=images or None,
                    width=request.width, height=request.height,
                    output_resolution=round((request.width * request.height) ** 0.5),
                    num_inference_steps=request.steps,
                    generator=torch.Generator("cuda").manual_seed(candidate_seed),
                    negative_prompt=request.negative_prompt or None,
                    true_cfg_scale=request.guidance_scale,
                    use_kv_cache=request.use_kv_cache,
                    callback_on_step_end=on_step,
                ).images[0]
            except Exception:
                # Diffusers' normal end-of-call hook cleanup is skipped on a
                # callback cancellation or OOM. Release offloaded components.
                try:
                    self.pipe.maybe_free_model_hooks()
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                raise
            if cancelled():
                raise Cancelled()
            if mask is not None:
                original = images[0].convert("RGBA").resize(image.size, Image.Resampling.LANCZOS)
                image = Image.composite(image.convert("RGBA"), original,
                                        mask.resize(image.size, Image.Resampling.NEAREST))
            name = f"{index + 1}.png"
            image.save(output / name, format="PNG")
            results.append({"file": name, "seed": candidate_seed, "width": image.width,
                            "height": image.height, "mode": image.mode})
        return results


class TaskStore:
    """Single GPU worker, durable receipts, cancellation and idempotent submission."""
    def __init__(self, root: Path, engine=None):
        self.root, self.engine = root, engine or QwenEngine()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.tasks: dict[str, dict] = {}
        self.pending = queue.Queue()
        self.worker = None
        for path in self.root.glob("*/task.json"):
            try:
                task = json.loads(path.read_text(encoding="utf-8"))
                if task["status"] not in TERMINAL:
                    task.update(status="failed", error="本地服务已重启，原任务中断；未自动重新生成")
                self.tasks[task["id"]] = task
                self.save(task)
            except (OSError, ValueError, KeyError):
                continue

    def save(self, task):
        directory = self.root / task["id"]
        directory.mkdir(exist_ok=True)
        temporary = directory / "task.tmp"
        temporary.write_text(json.dumps(task, ensure_ascii=False), encoding="utf-8")
        temporary.replace(directory / "task.json")

    def start(self):
        self.worker = threading.Thread(target=self.run, name="local-image-worker", daemon=True)
        self.worker.start()

    def close(self):
        with self.lock:
            for task in self.tasks.values():
                if task["status"] not in TERMINAL:
                    task["cancel_requested"] = True
        self.pending.put(None)
        if self.worker:
            self.worker.join(timeout=3)

    def submit(self, request: ImageRequest | None = None, kind="generate"):
        payload = request.model_dump() if request else {}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        with self.lock:
            request_id = payload.get("request_id")
            if request_id:
                for task in self.tasks.values():
                    if task.get("request_id") == request_id:
                        if task["request_hash"] != digest:
                            raise HTTPException(409, "request_id 已用于不同参数的任务")
                        return self.public(task)
            active = [t for t in self.tasks.values() if t["status"] not in TERMINAL]
            if len(active) >= 32:
                raise HTTPException(429, "本地生成队列已满")
            task = {"id": secrets.token_hex(16), "kind": kind, "status": "queued",
                    "progress": 0, "step": 0, "created_at": time.time(), "images": [],
                    "request_id": request_id, "request_hash": digest, "cancel_requested": False,
                    "parameters": {k: v for k, v in payload.items() if k not in {"images", "mask", "prompt", "negative_prompt"}}}
            self.tasks[task["id"]] = task
            self.save(task)
            self.pending.put((task["id"], request))
            return self.public(task)

    def public(self, task):
        result = {k: v for k, v in task.items() if k not in {"request_hash"}}
        result["images"] = [{**image, "url": f"/v1/tasks/{task['id']}/images/{i + 1}"}
                            for i, image in enumerate(task["images"])]
        result["queue_position"] = sum(1 for t in self.tasks.values()
                                       if t["status"] == "queued" and t["created_at"] < task["created_at"])
        return result

    def get(self, task_id):
        with self.lock:
            task = self.tasks.get(task_id)
            if task is None:
                raise HTTPException(404, "任务不存在")
            return self.public(task)

    def cancel(self, task_id):
        with self.lock:
            self.get(task_id)
            task = self.tasks[task_id]
            if task["status"] not in TERMINAL:
                task["cancel_requested"] = True
                if task["status"] == "queued":
                    task["status"] = "cancelled"
                self.save(task)
            return self.public(task)

    def run(self):
        while (item := self.pending.get()) is not None:
            task_id, request = item
            task = self.tasks[task_id]
            try:
                with self.lock:
                    if task["cancel_requested"]:
                        continue
                    task.update(status="loading", started_at=time.time())
                    self.save(task)
                if task["kind"] == "unload":
                    self.engine.unload()
                else:
                    self.engine.load()
                    if task["cancel_requested"]:
                        raise Cancelled()
                    if request:
                        with self.lock:
                            task["status"] = "running"
                            self.save(task)
                        def progress(value, step):
                            with self.lock:
                                task.update(progress=round(value, 4), step=step)
                        result = self.engine.generate(request, self.root / task_id, progress,
                                                      lambda: task["cancel_requested"])
                        with self.lock:
                            if task["cancel_requested"]:
                                raise Cancelled()
                            task["images"] = result
                with self.lock:
                    task.update(status="succeeded", progress=1)
            except Cancelled:
                with self.lock:
                    task.update(status="cancelled", error="用户已取消任务")
            except Exception as error:
                with self.lock:
                    task.update(status="failed", error=str(error))
            finally:
                with self.lock:
                    task["finished_at"] = time.time()
                    self.save(task)


def create_app(store: TaskStore | None = None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.store = store or TaskStore(DATA / "tasks")
        app.state.store.start()
        yield
        app.state.store.close()

    version = (ROOT / "VERSION").read_text(encoding="ascii").strip()
    app = FastAPI(title="Dreamer Local Image API", version=version, lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])

    @app.middleware("http")
    async def reject_cross_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            return JSONResponse({"detail": "仅接受同源页面或本机客户端请求"}, status_code=403)
        return await call_next(request)

    def tasks():
        return app.state.store

    @app.get("/", response_class=HTMLResponse)
    def home():
        return (ROOT / "index.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health():
        return {"service": "dreamer-local-image", "protocol": "dreamer-local-image-v1",
                "version": version,
                "model": MODEL_ID, **tasks().engine.status()}

    @app.get("/v1/capabilities")
    def get_capabilities():
        return capabilities()

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", **tasks().engine.status()}]}

    @app.post("/v1/model/load", status_code=202)
    def load():
        return tasks().submit(kind="load")

    @app.post("/v1/model/unload", status_code=202)
    def unload():
        return tasks().submit(kind="unload")

    @app.post("/v1/images/generations", status_code=202)
    @app.post("/v1/images/edits", status_code=202)
    def generate(request: ImageRequest):
        return tasks().submit(request)

    @app.get("/v1/tasks")
    def list_tasks():
        with tasks().lock:
            return {"data": [tasks().public(t) for t in sorted(tasks().tasks.values(),
                    key=lambda t: t["created_at"], reverse=True)[:50]]}

    @app.get("/v1/tasks/{task_id}")
    def get_task(task_id: str):
        return tasks().get(task_id)

    @app.delete("/v1/tasks/{task_id}")
    def cancel_task(task_id: str):
        return tasks().cancel(task_id)

    @app.get("/v1/tasks/{task_id}/images/{index}")
    def image(task_id: str, index: int):
        task = tasks().get(task_id)
        if task["status"] != "succeeded" or index < 1 or index > len(task["images"]):
            raise HTTPException(404, "结果图片尚不可用")
        return FileResponse(tasks().root / task_id / task["images"][index - 1]["file"], media_type="image/png")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("DREAMER_LOCAL_IMAGE_PORT", "8790")))
