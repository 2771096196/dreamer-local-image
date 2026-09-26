$ErrorActionPreference = 'Stop'
function Get-LocalSha256([string]$FilePath) {
    $hasher = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($FilePath)
    try { return [BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-', '') }
    finally { $stream.Dispose(); $hasher.Dispose() }
}
$packageRoot = $PSScriptRoot
$runtimeRoot = Join-Path $packageRoot 'runtime'
$pythonExe = Join-Path $runtimeRoot 'python.exe'
$settingsPath = Join-Path $packageRoot 'settings.json'
if (-not (Test-Path -LiteralPath $settingsPath)) {
    Copy-Item -LiteralPath (Join-Path $packageRoot 'settings.example.json') -Destination $settingsPath
}
$localSettings = Get-Content -LiteralPath $settingsPath -Raw -Encoding UTF8 | ConvertFrom-Json
$servicePort = [int]$localSettings.port
if ($servicePort -lt 1024 -or $servicePort -gt 65535) { throw 'Port must be between 1024 and 65535.' }
if ($localSettings.offload -notin @('sequential', 'model', 'none')) { throw 'Invalid offload setting.' }
try {
    $existingService = Invoke-RestMethod -Uri "http://127.0.0.1:$servicePort/health" -TimeoutSec 3
    if ($existingService.service -eq 'dreamer-local-image') {
        Write-Host "Service already running: http://127.0.0.1:$servicePort"
        exit 0
    }
    throw 'This port is occupied by a different service. Change port in settings.json.'
} catch [System.Net.WebException] {
    # A closed port is expected before the first start.
}
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw 'NVIDIA driver was not found. Install the NVIDIA driver and run Start.cmd again.'
}
& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
if ($LASTEXITCODE -ne 0) { throw 'NVIDIA driver check failed.' }

if (-not (Test-Path -LiteralPath $pythonExe)) {
    Write-Host 'Preparing isolated Python 3.12.10...'
    New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null
    $pythonArchive = Join-Path $runtimeRoot 'python.zip'
    Invoke-WebRequest -UseBasicParsing -Uri 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip' -OutFile $pythonArchive
    Expand-Archive -LiteralPath $pythonArchive -DestinationPath $runtimeRoot -Force
    Remove-Item -LiteralPath $pythonArchive
}
@('python312.zip', '.', '..', 'Lib\site-packages', 'import site') | Set-Content -LiteralPath (Join-Path $runtimeRoot 'python312._pth') -Encoding ASCII
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONUTF8 = '1'
$env:HF_HOME = Join-Path $packageRoot 'models'
if ($localSettings.hf_endpoint) { $env:HF_ENDPOINT = [string]$localSettings.hf_endpoint }
$env:DREAMER_LOCAL_IMAGE_PORT = [string]$servicePort
$env:DREAMER_LOCAL_IMAGE_OFFLOAD = [string]$localSettings.offload
$env:DREAMER_LOCAL_IMAGE_QUANTIZATION = if ($localSettings.quantization) { [string]$localSettings.quantization } else { 'none' }
$env:DREAMER_LOCAL_IMAGE_VAE_TILE_SIZE = if ($localSettings.vae_tile_size) { [string]$localSettings.vae_tile_size } else { '1024' }
$env:DREAMER_LOCAL_IMAGE_MODEL = [string]$localSettings.model
$env:DREAMER_LOCAL_IMAGE_DATA = Join-Path $packageRoot 'data'

$requirementsPath = Join-Path $packageRoot 'requirements.txt'
$quantRequirementsPath = Join-Path $packageRoot 'requirements-quant.txt'
$dependencyVersion = 'isolated-v2-torch-2.8.0-cu128-' + (Get-LocalSha256 $requirementsPath) + (Get-LocalSha256 $quantRequirementsPath)
$readyPath = Join-Path $runtimeRoot 'dependencies.ready'
$installedVersion = if (Test-Path -LiteralPath $readyPath) { (Get-Content -LiteralPath $readyPath -Raw).Trim() } else { '' }
if ($installedVersion -ne $dependencyVersion) {
    Write-Host 'Installing the NVIDIA inference runtime. The first start needs internet and several GB of disk space.'
    & $pythonExe -s -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('pip') else 1)"
    if ($LASTEXITCODE -ne 0) {
        $pipBootstrap = Join-Path $runtimeRoot 'get-pip.py'
        Invoke-WebRequest -UseBasicParsing -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile $pipBootstrap
        & $pythonExe -s $pipBootstrap --no-warn-script-location
        if ($LASTEXITCODE -ne 0) { throw 'pip bootstrap failed. Run Start.cmd again to retry.' }
    }
    & $pythonExe -s -m pip install --disable-pip-version-check --no-warn-script-location torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'CUDA runtime installation failed. Run Start.cmd again to retry.' }
    & $pythonExe -s -m pip install --disable-pip-version-check --no-warn-script-location -r $requirementsPath
    if ($LASTEXITCODE -ne 0) { throw 'Model dependency installation failed. Run Start.cmd again to retry.' }
    & $pythonExe -s -c "import torch; from diffusers import QwenImage21Pipeline; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name())"
    if ($LASTEXITCODE -ne 0) { throw 'Inference environment check failed.' }
    Set-Content -LiteralPath $readyPath -Value $dependencyVersion -Encoding ASCII
}
Write-Host "Local API: http://127.0.0.1:$servicePort"
Write-Host 'Open that address for status. The first Load Model / generation downloads model weights.'
Write-Host 'Keep this window open. Press Ctrl+C to stop the local service.'
& $pythonExe -s (Join-Path $packageRoot 'service.py')
exit $LASTEXITCODE
