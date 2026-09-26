Initial independent release of Dreamer Local Image.

- Windows/NVIDIA one-click bootstrap with an isolated Python/CUDA runtime.
- Qwen-Image-2.1 text-to-image, editing, multiple references and transparent PNG output.
- Mask-reference editing with explicit preservation compositing.
- NF4 quantization and component CPU offload for lower-memory configurations.
- Queued API tasks, progress, cancellation, idempotent submission and durable results.
- Resumable official model downloads with hash verification.
- Standalone status dashboard, OpenAPI docs and an HTTP protocol for any client.

See `docs/validation.md` for the exact locally tested hardware and results.
This is a bootstrap archive: first launch downloads Python/dependencies, and first model load downloads the official weights. Model weights and user data are not distributed.

Service code is MIT-licensed. Qwen-Image-2.1 retains its Qwen Research License; commercial model use requires separate authorization. See `NOTICE.md`.
