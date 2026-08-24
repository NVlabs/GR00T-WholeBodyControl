[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Apk,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9A-Fa-f]{64}$')]
    [string]$ExpectedSha256,

    [string]$LogPath = (Join-Path $env:TEMP 'xrobotoolkit-pico-adb-install.log'),

    [ValidateRange(1, 86400)]
    [int]$TimeoutSeconds = 14400
)

$ErrorActionPreference = 'Stop'
$packageName = 'com.xrobotoolkit.client.hardened'
$activityName = 'com.unity3d.player.UnityPlayerActivity'

function Write-InstallLog {
    param([string]$Message)
    $line = '{0:o} {1}' -f (Get-Date), $Message
    Add-Content -LiteralPath $LogPath -Value $line
}

if (-not (Test-Path -LiteralPath $Apk -PathType Leaf)) {
    throw "APK not found: $Apk"
}

$actualSha256 = (Get-FileHash -LiteralPath $Apk -Algorithm SHA256).Hash
if ($actualSha256 -ne $ExpectedSha256.ToUpperInvariant()) {
    throw "APK SHA256 mismatch: expected $ExpectedSha256, got $actualSha256"
}

$adb = (Get-Command adb -ErrorAction Stop).Source
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
Write-InstallLog "watching for PICO; apk_sha256=$actualSha256"

while ((Get-Date) -lt $deadline) {
    $deviceLines = & $adb devices 2>&1 | Select-Object -Skip 1
    foreach ($line in $deviceLines) {
        if ($line -notmatch '^([^\s]+)\s+device$') {
            continue
        }

        $serial = $Matches[1]
        $manufacturer = ((& $adb -s $serial shell getprop ro.product.manufacturer 2>$null) -join '').Trim()
        $brand = ((& $adb -s $serial shell getprop ro.product.brand 2>$null) -join '').Trim()
        $model = ((& $adb -s $serial shell getprop ro.product.model 2>$null) -join '').Trim()
        $identity = "$manufacturer $brand $model"
        if ($identity -notmatch '(?i)pico|picoxr|pvr') {
            Write-InstallLog "ignored non-PICO adb device serial=$serial identity=$identity"
            continue
        }

        Write-InstallLog "PICO found serial=$serial identity=$identity"
        $installOutput = (& $adb -s $serial install -r $Apk 2>&1) -join "`n"
        Write-InstallLog "adb install output: $installOutput"
        if ($LASTEXITCODE -ne 0 -or $installOutput -notmatch '(?m)^Success\s*$') {
            throw "ADB install failed for $serial. See $LogPath"
        }

        $packagePath = ((& $adb -s $serial shell pm path $packageName 2>&1) -join '').Trim()
        if ($LASTEXITCODE -ne 0 -or $packagePath -notmatch '^package:') {
            throw "Installed package verification failed for $serial. See $LogPath"
        }

        $component = "$packageName/$activityName"
        $startOutput = (& $adb -s $serial shell am start -n $component 2>&1) -join "`n"
        Write-InstallLog "adb start output: $startOutput"
        if ($LASTEXITCODE -ne 0 -or $startOutput -match '(?i)error|exception') {
            throw "ADB app launch failed for $serial. See $LogPath"
        }

        Write-InstallLog "success serial=$serial package=$packageName"
        exit 0
    }

    Start-Sleep -Seconds 2
}

throw "Timed out waiting for an authorized PICO ADB device after $TimeoutSeconds seconds"
