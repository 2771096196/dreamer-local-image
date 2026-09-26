# Third-party models and packages

The MIT license in this repository applies to the independently written API
service, launcher, dashboard and tests. It does not relicense any model,
model output, or third-party dependency.

Qwen-Image-2.1 is developed by the Qwen team. The default model is obtained
directly from its official repository at runtime; weights are not included
in this repository or its release archives.

- Model: https://huggingface.co/Qwen/Qwen-Image-2.1
- License: https://github.com/QwenLM/Qwen-Image-2.1/blob/main/LICENSE

At the time this integration was written, the Qwen Research License permits
non-commercial research/evaluation and requires separate authorization for
commercial use. Review the current model license before downloading or using
the weights. Installing this MIT-licensed service does not grant commercial
rights to the model.

Python, PyTorch, Diffusers, Transformers, Accelerate, FastAPI, Uvicorn, Pillow
and other dependencies retain their respective licenses and notices. The
bootstrap release downloads these packages from their official sources.
