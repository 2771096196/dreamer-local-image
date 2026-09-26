# Local validation

This document records measured results. API unit tests do not constitute real-model validation.

## Environment

- Date: 2026-09-26
- OS: Windows x64
- GPU: NVIDIA GeForce RTX 3060, 12GB; driver 591.86
- System RAM: 32GB
- Python: isolated portable 3.12.10 installed by `Start.cmd`
- PyTorch: 2.8.0+cu128; torchvision 0.23.0+cu128
- Diffusers: commit `e0abab83b5df05de9e7abd788643c1a7c1e42e28`
- Transformers: 5.17.0; bitsandbytes 0.50.2
- Official model revision: `790c92633540aa0cb11d9abf19eb46d861714758`

## Completed checks

- Fresh portable Python, pip and CUDA runtime installation through the actual Windows launcher.
- CUDA detects the RTX 3060.
- NF4 CUDA kernel evaluation and CPU → CUDA round-trip: finite output, zero difference in the kernel check.
- API/queue/download tests: 27 passed, using an explicitly fake engine.
- Default HTTP service starts and serves its status dashboard.
- Chinese/space directory launch: passed, using the installed portable runtime.
- Explicit `-s` isolation: `pip check` reports no broken requirements; user-global Python packages are excluded.
- Real local HTTP ranged-download tests cover resume, servers ignoring Range, segmented resume and official hash verification.
- Separate Dreamer consumer integration: 5 checks passed; those tests remain outside this repository.

## Real model acceptance

All cases below completed on the default **NF4 + model CPU offload + 1024px VAE tile** profile. Timings are local API end-to-end timings, including polling and downloading the PNG; they are examples from this machine, not a speed guarantee. Peak VRAM is PyTorch's per-task allocated-memory peak, not all applications' combined GPU usage.

Model loading and quantization: 49.75 seconds after weights were downloaded.

| Case | Size | Steps | Images | API seconds | Peak allocated MiB | Result |
|---|---|---:|---:|---:|---:|---|
| text-to-image | 1024×1024 | 40 | 1 | 146.27 | 7399 | passed |
| image-edit | 512×512 | 12 | 1 | 18.08 | 7287 | passed |
| multi-reference | 512×512 | 12 | 1 | 18.05 | 7289 | passed |
| transparent | 512×512 | 12 | 1 | 14.06 | 6139 | passed |
| mask-edit | 512×512 | 12 | 1 | 18.06 | 7291 | passed |
| batch | 512×512 | 12 | 2 | 28.08 | 6126 | passed |
| seed-a | 512×512 | 4 | 1 | 8.06 | 6125 | passed |
| seed-b | 512×512 | 4 | 1 | 8.08 | 6125 | passed |
| negative-and-no-cache | 512×512 | 4 | 1 | 12.03 | 6125 | passed |
| ten-references | 256×256 | 2 | 1 | 8.08 | 6894 | passed |
| 2k-memory-smoke | 2048×2048 | 1 | 1 | 34.12 | 7440 | passed |

Additional real-service checks:

- The two identical-seed PNGs had identical SHA-256 hashes.
- Transparent output contained both transparent and visible pixels in its real alpha channel.
- The masked edit retained every pixel outside the editable mask exactly.
- Re-submitting an identical request ID returned the existing task without another generation.
- Cancelling during actual denoising settled as cancelled with no output images; the next generation succeeded.
- Restarting the Windows process preserved previous successful task receipts and exact PNG hashes.
- The real Dreamer HTTP adapter generated and retrieved a 1344×768 PNG using a 16:9 / 1K request.
- Width and height are constrained to the pipeline’s true 32px latent grid. An invalid 1360px width was rejected by the live API with HTTP 422 instead of silently shrinking the output.

### Visual inspection and limits

The initial 256px VAE tiling setting created visible seam-like colored stripes on flat backgrounds. A same-prompt/seed comparison at 1024px tiling removed those artifacts; the published default keeps 1K output untiled. Representative original outputs are in `docs/examples`.

The **2K case is a one-step memory/API smoke test**, not a 2K quality benchmark or long-run stability claim. The 10-reference case uses small synthetic fixtures. Other GPUs, larger reference images, longer prompts and unquantized profiles have not been validated here. NF4 has a quality/precision tradeoff compared with BF16.

The service is a loopback-only single-GPU queue. Mask editing is reference-guided generation plus preservation compositing, not native latent inpainting.

Large weights, local settings, raw logs and generated task data are excluded from Git and release archives. `scripts/qa_gpu.py` reproduces this acceptance matrix using synthetic inputs only.
