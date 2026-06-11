param(
    [switch]$IfNeeded
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$RequiredPaths = @(
    @{ Name = "Python virtual environment"; Path = Join-Path $Root ".venv\Scripts\python.exe" },
    @{ Name = "HYPIR source checkout"; Path = Join-Path $Root "external\HYPIR\HYPIR\enhancer\sd2.py" },
    @{ Name = "SUPIR source checkout"; Path = Join-Path $Root "external\SUPIR\README.md" },
    @{ Name = "HYPIR SD2 weights"; Path = Join-Path $Root "models\hypir\HYPIR_sd2.pth" },
    @{ Name = "Stable Diffusion 2.1 base model"; Path = Join-Path $Root "models\stable-diffusion-2-1-base\model_index.json" }
)

function Refresh-ProcessPath {
    $pathParts = @(
        [Environment]::GetEnvironmentVariable("Path", "User"),
        [Environment]::GetEnvironmentVariable("Path", "Machine"),
        (Join-Path $env:USERPROFILE ".local\bin"),
        (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links")
    ) | Where-Object { $_ }

    $env:Path = ($pathParts -join ";")
}

function Get-ToolPath {
    param([string]$Name)

    $command = Get-Command $Name -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $command) {
        return $command.Source
    }

    if ($Name -eq "uv") {
        $candidates = @(
            (Join-Path $env:USERPROFILE ".local\bin\uv.exe"),
            (Join-Path $env:LOCALAPPDATA "Programs\uv\uv.exe")
        )

        foreach ($candidate in $candidates) {
            if (Test-Path -LiteralPath $candidate) {
                return $candidate
            }
        }
    }

    return $null
}

function Get-MissingItems {
    $missing = @()

    foreach ($item in $RequiredPaths) {
        if (!(Test-Path -LiteralPath $item.Path)) {
            $missing += $item.Name
        }
    }

    foreach ($tool in @("ffmpeg", "ffprobe")) {
        if (!(Get-ToolPath $tool)) {
            $missing += "$tool command"
        }
    }

    return $missing
}

function Invoke-Native {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$StepName
    )

    Write-Host $StepName
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE."
    }
}

function Write-ManualRecovery {
    param([string]$Reason)

    Write-Host ""
    Write-Host "CleanVideo setup could not finish automatically." -ForegroundColor Yellow
    Write-Host "Reason: $Reason"
    Write-Host ""
    Write-Host "Manual recovery steps:"
    Write-Host "  1. Install uv:"
    Write-Host "     powershell -NoProfile -ExecutionPolicy Bypass -Command `"irm https://astral.sh/uv/install.ps1 | iex`""
    Write-Host "  2. Install FFmpeg:"
    Write-Host "     winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements"
    Write-Host "  3. Download HYPIR source:"
    Write-Host "     https://github.com/XPixelGroup/HYPIR/archive/HEAD.zip"
    Write-Host "     Extract the inner folder to:"
    Write-Host "     $Root\external\HYPIR"
    Write-Host "  4. Download SUPIR source:"
    Write-Host "     https://github.com/Fanghua-Yu/SUPIR/archive/HEAD.zip"
    Write-Host "     Extract the inner folder to:"
    Write-Host "     $Root\external\SUPIR"
    Write-Host "  5. Download HYPIR weights:"
    Write-Host "     https://huggingface.co/lxq007/HYPIR/resolve/main/HYPIR_sd2.pth"
    Write-Host "     Save the file as:"
    Write-Host "     $Root\models\hypir\HYPIR_sd2.pth"
    Write-Host "  6. Download the Stable Diffusion 2.1 base model mirror:"
    Write-Host "     https://huggingface.co/Manojb/stable-diffusion-2-1-base/tree/main"
    Write-Host "     Put these folders/files under:"
    Write-Host "     $Root\models\stable-diffusion-2-1-base"
    Write-Host "     Required: scheduler, tokenizer, text_encoder, unet, vae, model_index.json"
    Write-Host "  7. Run Start-CleanVideo.cmd again."
    Write-Host ""
}

function Ensure-Uv {
    Refresh-ProcessPath
    $uv = Get-ToolPath "uv"
    if ($uv) {
        return $uv
    }

    Write-Host "uv was not found. Installing uv with the official Astral installer..."
    try {
        Invoke-RestMethod -Uri "https://astral.sh/uv/install.ps1" | Invoke-Expression
    } catch {
        throw "Could not install uv automatically. $($_.Exception.Message)"
    }

    Refresh-ProcessPath
    $uv = Get-ToolPath "uv"
    if (!$uv) {
        throw "uv was installed, but it is not visible in this terminal PATH yet. Close the terminal and run Start-CleanVideo.cmd again, or install uv manually."
    }

    return $uv
}

function Ensure-FFmpeg {
    Refresh-ProcessPath
    if ((Get-ToolPath "ffmpeg") -and (Get-ToolPath "ffprobe")) {
        return
    }

    $winget = Get-ToolPath "winget"
    if ($winget) {
        Write-Host "FFmpeg was not found. Installing FFmpeg with winget..."
        Invoke-Native `
            -FilePath $winget `
            -Arguments @("install", "--id", "Gyan.FFmpeg", "-e", "--accept-source-agreements", "--accept-package-agreements") `
            -StepName "Installing FFmpeg"
        Refresh-ProcessPath
    }

    if (!(Get-ToolPath "ffmpeg") -or !(Get-ToolPath "ffprobe")) {
        throw "FFmpeg or ffprobe is missing. Install FFmpeg and make sure both commands are on PATH."
    }
}

function Install-GitHubRepository {
    param(
        [string]$Name,
        [string]$GitUrl,
        [string]$ZipUrl,
        [string]$Destination,
        [string]$ProbePath
    )

    if (Test-Path -LiteralPath $ProbePath) {
        return
    }

    if (Test-Path -LiteralPath $Destination) {
        throw "$Name directory already exists, but the expected file is missing: $ProbePath. Move that directory aside or replace it with a complete checkout."
    }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null

    $git = Get-ToolPath "git"
    if ($git) {
        Invoke-Native -FilePath $git -Arguments @("clone", $GitUrl, $Destination) -StepName "Cloning $Name"
        return
    }

    Write-Host "Git was not found. Downloading $Name as a GitHub ZIP archive..."
    $tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("cleanvideo-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null

    try {
        $archivePath = Join-Path $tempRoot "$Name.zip"
        $extractPath = Join-Path $tempRoot "extract"
        Invoke-WebRequest -Uri $ZipUrl -OutFile $archivePath
        Expand-Archive -LiteralPath $archivePath -DestinationPath $extractPath -Force

        $source = Get-ChildItem -LiteralPath $extractPath -Directory | Select-Object -First 1
        if ($null -eq $source) {
            throw "GitHub ZIP archive for $Name did not contain a top-level source directory."
        }

        Move-Item -LiteralPath $source.FullName -Destination $Destination
    } finally {
        if (Test-Path -LiteralPath $tempRoot) {
            $resolvedTemp = (Resolve-Path -LiteralPath $tempRoot).Path
            $safeTempRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
            if ($resolvedTemp.StartsWith($safeTempRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
                Remove-Item -LiteralPath $resolvedTemp -Recurse -Force
            }
        }
    }
}

function Show-CudaStatus {
    $python = Join-Path $Root ".venv\Scripts\python.exe"
    if (!(Test-Path -LiteralPath $python)) {
        return
    }

    & $python -c "import torch, sys; print('PyTorch ' + torch.__version__); print('CUDA available: ' + str(torch.cuda.is_available())); print('CUDA runtime: ' + str(torch.version.cuda)); print('GPU: ' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not detected')); sys.exit(0 if torch.cuda.is_available() else 2)"
    if ($LASTEXITCODE -eq 2) {
        Write-Host "WARNING: CUDA is not available. CleanVideo needs an NVIDIA CUDA GPU for practical HYPIR export." -ForegroundColor Yellow
    } elseif ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Could not verify CUDA status." -ForegroundColor Yellow
    }
}

try {
    if ($IfNeeded) {
        $missing = @(Get-MissingItems)
        if ($missing.Count -eq 0) {
            Write-Host "CleanVideo setup is already complete."
            exit 0
        }

        Write-Host "CleanVideo setup is missing: $($missing -join ', ')"
    }

    Ensure-FFmpeg

    $uv = Ensure-Uv
    $python = Join-Path $Root ".venv\Scripts\python.exe"
    if (!(Test-Path -LiteralPath $python)) {
        Invoke-Native -FilePath $uv -Arguments @("python", "install", "3.10") -StepName "Installing Python 3.10"
        Invoke-Native -FilePath $uv -Arguments @("venv", "--python", "3.10", ".venv") -StepName "Creating Python virtual environment"
        Invoke-Native -FilePath $python -Arguments @("-m", "ensurepip", "--upgrade") -StepName "Bootstrapping pip"
    }

    Invoke-Native -FilePath $python -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel") -StepName "Upgrading pip tooling"
    Invoke-Native -FilePath $python -Arguments @("-m", "pip", "install", "-r", "requirements-inference.txt") -StepName "Installing Python dependencies"

    Install-GitHubRepository `
        -Name "HYPIR" `
        -GitUrl "https://github.com/XPixelGroup/HYPIR.git" `
        -ZipUrl "https://github.com/XPixelGroup/HYPIR/archive/HEAD.zip" `
        -Destination (Join-Path $Root "external\HYPIR") `
        -ProbePath (Join-Path $Root "external\HYPIR\HYPIR\enhancer\sd2.py")

    Install-GitHubRepository `
        -Name "SUPIR" `
        -GitUrl "https://github.com/Fanghua-Yu/SUPIR.git" `
        -ZipUrl "https://github.com/Fanghua-Yu/SUPIR/archive/HEAD.zip" `
        -Destination (Join-Path $Root "external\SUPIR") `
        -ProbePath (Join-Path $Root "external\SUPIR\README.md")

    Invoke-Native -FilePath $python -Arguments @("scripts\download_models.py") -StepName "Downloading model files"

    $missingAfterSetup = @(Get-MissingItems)
    if ($missingAfterSetup.Count -gt 0) {
        throw "Setup finished, but required items are still missing: $($missingAfterSetup -join ', ')"
    }

    Show-CudaStatus
    Write-Host "CleanVideo setup complete."
} catch {
    Write-ManualRecovery -Reason $_.Exception.Message
    exit 1
}
