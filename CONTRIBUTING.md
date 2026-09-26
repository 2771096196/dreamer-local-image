# Contributing

Use Python 3.12. For API development without downloading models:

```shell
python -m venv .venv
# Activate the environment, then:
python -m pip install -r requirements-dev.txt
python -m pytest
```

Tests use an explicitly fake inference engine for deterministic queue/API checks.
They do not claim GPU or image-quality validation. Real GPU acceptance results
belong in `docs/validation.md`, with model revision, device, settings and measured
results. Do not commit model weights, user images, task data, caches, credentials
or machine-specific settings.

The wire protocol is `dreamer-local-image-v1`. Keep clients decoupled from this
repository: they submit JSON and query task receipts, never import the worker.
Publish changes to parameters/capabilities alongside the OpenAPI schema.
