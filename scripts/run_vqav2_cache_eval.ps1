param(
    [string]$Dataset = "data/datasets/vqav2_cache_eval/vqav2_cache_100img_3q_300.jsonl",
    [string]$ExperimentsDir = "data/experiments/vqav2_cache_eval_runs",
    [string]$RunPrefix = "vqav2_cache_100img_3q"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Dataset)) {
    throw "Dataset not found: $Dataset"
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$cacheOnRun = "${RunPrefix}_${timestamp}_cache_on"
$runtimeConfigDir = Join-Path $ExperimentsDir "_runtime_configs"
New-Item -ItemType Directory -Force -Path $runtimeConfigDir | Out-Null

$cacheOnConfig = Join-Path $runtimeConfigDir "${cacheOnRun}.yaml"

@"
vlm:
  enabled: true
  skip_if_no_image: true
  cache_enabled: true
  caption_mode: image_caption

fallback:
  enabled: false

cache:
  enabled: true
  cache_dir: $ExperimentsDir/$cacheOnRun/cache
  cache_vlm: true
  cache_router: false
  cache_llm: false

runtime:
  experiment_version: v2
  experiment_mode: vqav2_cache_on
"@ | Set-Content -LiteralPath $cacheOnConfig -Encoding UTF8

Write-Host "Running cache-on evaluation: $cacheOnRun"
python scripts/run_experiment.py `
  --mode difficulty_cache `
  --dataset $Dataset `
  --run-id $cacheOnRun `
  --experiments-dir $ExperimentsDir `
  --override-config $cacheOnConfig `
  --no-resume

Write-Host "Cache-on results: $ExperimentsDir/$cacheOnRun/difficulty_cache/results.jsonl"
Write-Host "Runtime config:   $cacheOnConfig"
Write-Host "Cache-off baseline should be estimated later from the cache-on log by charging repeated-image rows the first observed uncached VLM latency for the same image_id."
