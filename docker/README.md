# AdaRoute-MM Docker 8GB Experiment

This Docker path is for Windows VSCode editing plus containerized execution. Do not open VSCode inside the containers.

## Services

- `ollama`: official `ollama/ollama` image, port `11434`, persistent model volume `ollama_models`.
- `adaroute`: Python runtime built from `docker/Dockerfile`, with `mem_limit: 8g`, `PYTHONPATH=/app`, and `OLLAMA_BASE_URL=http://ollama:11434`.

## Quick Start On Windows

From the project root:

```powershell
.\run_docker_experiment.ps1
```

Use GPU when Docker Desktop and NVIDIA support are already available:

```powershell
.\run_docker_experiment.ps1 -Gpu
```

The script starts Ollama, pulls the configured models, builds the AdaRoute-MM image, and runs the six-mode 100-sample v3_2 experiment. Results are written to:

```text
results/docker_8gb/<run-id>/
```

## Model Configuration

Defaults are set through environment variables in `docker-compose.yml` and can be overridden from PowerShell:

```powershell
.\run_docker_experiment.ps1 `
  -LargeModel "gemma3n:e2b" `
  -MediumModel "phi3:latest" `
  -SampleSize 100
```

The project reads model names from config first, then Docker overrides these model keys through environment variables:

- `ADAROUTE_ROUTER_MODEL`
- `ADAROUTE_SMALL_MODEL`
- `ADAROUTE_MEDIUM_MODEL`
- `ADAROUTE_LARGE_MODEL`
- `ADAROUTE_VLM_MODEL`

`OLLAMA_BASE_URL` is resolved in this order: environment variable, config file, then `http://localhost:11434`.

## Manual Commands

Start Ollama:

```powershell
docker compose up -d ollama
```

Pull models into the persistent Ollama volume:

```powershell
docker compose exec ollama sh /scripts/pull_models.sh
```

Build and run the AdaRoute-MM experiment container:

```powershell
docker compose build adaroute
docker compose run --rm adaroute
```

For GPU:

```powershell
docker compose -f docker-compose.yml -f docker/docker-compose.gpu.yml up -d ollama
docker compose -f docker-compose.yml -f docker/docker-compose.gpu.yml run --rm adaroute
```
