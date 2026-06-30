param(
    [string]$EnvBase = "$HOME\Envs",
    [string[]]$EnvNames = @("mlchemenv312", "mlchemenv313", "mlchemenv314"),
    [string[]]$PytestArgs = @("-vv", "tests"),
    [switch]$FailFast
)

$ErrorActionPreference = "Stop"

$results = @()

foreach ($name in $EnvNames) {
    $pythonPath = Join-Path $EnvBase "$name\Scripts\python.exe"

    if (-not (Test-Path $pythonPath)) {
        Write-Host "=== $name ===" -ForegroundColor Yellow
        Write-Host "Missing interpreter: $pythonPath" -ForegroundColor Red
        $results += [pscustomobject]@{
            Environment = $name
            Status = "missing"
            ExitCode = 127
        }
        if ($FailFast) {
            exit 127
        }
        continue
    }

    Write-Host "=== $name ===" -ForegroundColor Cyan
    & $pythonPath -m pytest @PytestArgs
    $exitCode = $LASTEXITCODE

    if ($exitCode -eq 0) {
        $status = "pass"
        Write-Host "Result: pass" -ForegroundColor Green
    }
    else {
        $status = "fail"
        Write-Host "Result: fail (exit $exitCode)" -ForegroundColor Red
    }

    $results += [pscustomobject]@{
        Environment = $name
        Status = $status
        ExitCode = $exitCode
    }

    if ($FailFast -and $exitCode -ne 0) {
        exit $exitCode
    }
}

Write-Host ""
Write-Host "Summary" -ForegroundColor White
$results | Format-Table -AutoSize

if ($results | Where-Object { $_.Status -eq "fail" -or $_.Status -eq "missing" }) {
    exit 1
}

exit 0
