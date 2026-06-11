param(
    [switch]$Gpu,
    [switch]$SkipPullModels,
    [switch]$ResumeExisting,
    [string]$RunId = "",
    [int]$SampleSize = 100,
    [string]$Dataset = "data/datasets/v3_1_text_fusion/fusion_v3_1_1000_200-200-400-100-100.jsonl",
    [string]$RouterModel = "qwen2.5:1.5b",
    [string]$SmallModel = "qwen2.5:1.5b",
    [string]$MediumModel = "phi3:latest",
    [string]$LargeModel = "sam860/gemma3n:e2b-Q3_K_XL",
    [string]$VlmModel = "moondream:latest"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$ResultsRoot = Join-Path $ProjectRoot "results/docker_8gb"
New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null

if (-not $RunId) {
    $RunId = "docker_8gb_{0}" -f (Get-Date -Format "yyyyMMdd-HHmmss")
}

$RequestedRunId = $RunId
$suffix = 1
while ((Test-Path (Join-Path $ResultsRoot $RunId)) -and (-not $ResumeExisting)) {
    $RunId = "{0}_{1:D2}" -f $RequestedRunId, $suffix
    $suffix += 1
}

$ResultPath = Join-Path $ResultsRoot $RunId
New-Item -ItemType Directory -Force -Path $ResultPath | Out-Null
$TranscriptPath = Join-Path $ResultPath "docker_run.log"

$env:ADAROUTE_RUN_ID = $RunId
$env:ADAROUTE_SAMPLE_SIZE = [string]$SampleSize
$env:ADAROUTE_DATASET = $Dataset
$env:ADAROUTE_ROUTER_MODEL = $RouterModel
$env:ADAROUTE_SMALL_MODEL = $SmallModel
$env:ADAROUTE_MEDIUM_MODEL = $MediumModel
$env:ADAROUTE_LARGE_MODEL = $LargeModel
$env:ADAROUTE_VLM_MODEL = $VlmModel

$composeFiles = @("-f", "docker-compose.yml")
if ($Gpu) {
    $composeFiles += @("-f", "docker/docker-compose.gpu.yml")
}

$ContainerName = ("adaroute-mm-run-{0}" -f ($RunId -replace '[^a-zA-Z0-9_.-]', '-')).ToLowerInvariant()

function Invoke-Compose {
    param([string[]]$Arguments)
    & docker @("compose") @composeFiles @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

function Invoke-Docker {
    param([string[]]$Arguments)
    & docker @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "docker $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

function Test-ContainerExists {
    param([string]$Name)
    $null = & docker @("inspect", $Name) 2>$null
    return $LASTEXITCODE -eq 0
}

function Get-ContainerRunning {
    param([string]$Name)
    $value = & docker @("inspect", "-f", "{{.State.Running}}", $Name)
    if ($LASTEXITCODE -ne 0) {
        throw "docker inspect failed for $Name"
    }
    return $value.Trim() -eq "true"
}

function Get-ContainerExitCode {
    param([string]$Name)
    $value = & docker @("inspect", "-f", "{{.State.ExitCode}}", $Name)
    if ($LASTEXITCODE -ne 0) {
        throw "docker inspect failed for $Name"
    }
    return [int]$value.Trim()
}

$TranscriptStarted = $false
try {
    Start-Transcript -Path $TranscriptPath -Force | Out-Null
    $TranscriptStarted = $true

    Write-Host "Run ID: $RunId"
    Write-Host "Log file: $TranscriptPath"
    Write-Host "Starting Ollama service..."
    Invoke-Compose @("up", "-d", "ollama")

    if (-not $SkipPullModels) {
        Write-Host "Pulling configured Ollama models into the persistent Docker volume..."
        Invoke-Compose @("exec", "-T", "ollama", "sh", "/scripts/pull_models.sh")
    }

    Write-Host "Building AdaRoute-MM experiment image..."
    Invoke-Compose @("build", "adaroute")

    Write-Host "Running AdaRoute-MM 8GB experiment: $RunId"
    if (Test-ContainerExists $ContainerName) {
        if (Get-ContainerRunning $ContainerName) {
            Write-Host "Container already running: $ContainerName"
        }
        elseif ($ResumeExisting) {
            Write-Host "Removing stopped resume container: $ContainerName"
            Invoke-Docker @("rm", $ContainerName)
            Invoke-Compose @("run", "-d", "--name", $ContainerName, "adaroute")
        }
        else {
            throw "Container already exists: $ContainerName. Use -ResumeExisting or remove it manually."
        }
    }
    else {
        Invoke-Compose @("run", "-d", "--name", $ContainerName, "adaroute")
    }

    while (Get-ContainerRunning $ContainerName) {
        Start-Sleep -Seconds 15
    }
    $ExitCode = Get-ContainerExitCode $ContainerName
    Write-Host "AdaRoute container exit code: $ExitCode"
    Write-Host "Container logs:"
    & docker @("logs", $ContainerName)
    Invoke-Docker @("rm", $ContainerName)
    if ($ExitCode -ne 0) {
        throw "AdaRoute container failed with exit code $ExitCode"
    }

    Write-Host "Experiment complete. Results saved to: $ResultPath"
}
finally {
    if ($TranscriptStarted) {
        Stop-Transcript | Out-Null
    }
}
