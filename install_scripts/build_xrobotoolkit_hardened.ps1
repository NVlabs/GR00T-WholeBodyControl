[CmdletBinding()]
param(
    [string]$SourceRoot = (Join-Path $env:TEMP 'groot-wbc-xrobotoolkit-hardened'),
    [string]$UnityEditor = 'C:\Program Files\Unity\Hub\Editor\2022.3.16f1\Editor\Unity.exe',
    [string]$OutputApk = (Join-Path $env:TEMP 'XRoboToolkit-PICO-1.1.1-hardened.1.apk'),
    [string]$SdkRoot,
    [string]$NdkRoot,
    [string]$JdkRoot,
    [string]$AdbPath,
    [switch]$AllowUnityBundledLegacyNdkForDiagnostic,
    [switch]$AllowUnityBundledLegacyJdkForDiagnostic,
    [switch]$InstallToConnectedPico
)

$ErrorActionPreference = 'Stop'
$ExpectedCommit = '9f775b535d781618bd2bb7ef8d6c414c0531387c'
$Repository = 'https://github.com/XR-Robotics/XRoboToolkit-Unity-Client.git'
$PatchPath = Join-Path $PSScriptRoot 'xrobotoolkit_pico_health_v1.patch'

function Invoke-Checked {
    param(
        [Parameter(Mandatory)]
        [string]$FilePath,
        [Parameter(ValueFromRemainingArguments)]
        [string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path -LiteralPath $PatchPath -PathType Leaf)) {
    throw "Hardened client patch missing: $PatchPath"
}
if (-not (Test-Path -LiteralPath $UnityEditor -PathType Leaf)) {
    throw "Unity 2022.3.16f1 missing: $UnityEditor"
}
$AndroidPlayer = Join-Path (Split-Path (Split-Path $UnityEditor -Parent) -Parent) `
    'Editor\Data\PlaybackEngines\AndroidPlayer'
if (-not (Test-Path -LiteralPath $AndroidPlayer -PathType Container)) {
    throw "Unity Android Build Support missing under $AndroidPlayer"
}
$EmbeddedSdk = Join-Path $AndroidPlayer 'SDK'
if ([string]::IsNullOrWhiteSpace($SdkRoot)) {
    $SdkRoot = $EmbeddedSdk
}
$SdkRoot = [System.IO.Path]::GetFullPath($SdkRoot)
$SdkManagerCandidates = @(
    (Join-Path $SdkRoot 'cmdline-tools\latest\bin\sdkmanager.bat'),
    (Join-Path $SdkRoot 'tools\bin\sdkmanager.bat')
)
if (-not ($SdkManagerCandidates | Where-Object {
    Test-Path -LiteralPath $_ -PathType Leaf
})) {
    throw "Android SDK manager missing under $SdkRoot"
}
foreach ($SdkTool in @(
    'platform-tools\adb.exe',
    'platform-tools\source.properties',
    'build-tools\30.0.2\aapt2.exe',
    'platforms\android-30\android.jar'
)) {
    $SdkToolPath = Join-Path $SdkRoot $SdkTool
    if (-not (Test-Path -LiteralPath $SdkToolPath -PathType Leaf)) {
        throw "Android SDK component missing: $SdkToolPath"
    }
}
$PlatformToolsProperties = Get-Content -Raw -LiteralPath (
    Join-Path $SdkRoot 'platform-tools\source.properties'
)
if (
    $PlatformToolsProperties -notmatch '(?m)^Pkg\.Revision\s*=\s*([0-9.]+)\s*$' -or
    [version]$Matches[1] -lt [version]'32.0.0'
) {
    throw "Unity 2022 Android build requires SDK Platform Tools 32.0.0 or newer"
}
$EmbeddedNdk = Join-Path $AndroidPlayer 'NDK'
if ([string]::IsNullOrWhiteSpace($NdkRoot)) {
    $NdkRoot = $EmbeddedNdk
}
$NdkRoot = [System.IO.Path]::GetFullPath($NdkRoot)
$NdkProperties = Join-Path $NdkRoot 'source.properties'
$NdkClang = Join-Path $NdkRoot 'toolchains\llvm\prebuilt\windows-x86_64\bin\clang.exe'
if (
    -not (Test-Path -LiteralPath $NdkProperties -PathType Leaf) -or
    -not (Test-Path -LiteralPath $NdkClang -PathType Leaf)
) {
    throw "Complete Android NDK missing under $NdkRoot"
}
$NdkVersion = Get-Content -Raw -LiteralPath $NdkProperties
if ($NdkVersion -notmatch '(?m)^Pkg\.Revision\s*=\s*23\.1\.7779620\s*$') {
    if (
        $AllowUnityBundledLegacyNdkForDiagnostic -and
        $NdkVersion -match '(?m)^Pkg\.Revision\s*=\s*21\.3\.6528147\s*$'
    ) {
        Write-Warning (
            'Using Unity-bundled NDK r21d for a diagnostic-only Pico build. ' +
            'This artifact must not be treated as the reproducible r23b release build.'
        )
    } else {
        throw "Unity 2022.3.16f1 requires Android NDK r23b (23.1.7779620)"
    }
}
$EmbeddedJdk = Join-Path $AndroidPlayer 'OpenJDK'
if ([string]::IsNullOrWhiteSpace($JdkRoot)) {
    $JdkRoot = $EmbeddedJdk
}
$JdkRoot = [System.IO.Path]::GetFullPath($JdkRoot)
$Java = Join-Path $JdkRoot 'bin\java.exe'
$Javac = Join-Path $JdkRoot 'bin\javac.exe'
if (
    -not (Test-Path -LiteralPath $Java -PathType Leaf) -or
    -not (Test-Path -LiteralPath $Javac -PathType Leaf)
) {
    throw "Complete JDK missing under $JdkRoot"
}
$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$JavaVersion = (& $Java -version 2>&1 | Out-String)
$JavaExitCode = $LASTEXITCODE
$ErrorActionPreference = $PreviousErrorActionPreference
if ($JavaExitCode -ne 0 -or $JavaVersion -notmatch 'version "11(?:\.|")') {
    if (
        $AllowUnityBundledLegacyJdkForDiagnostic -and
        $JavaExitCode -eq 0 -and
        $JavaVersion -match '64-Bit' -and
        $JavaVersion -match 'version "1\.8\.'
    ) {
        Write-Warning (
            'Using Unity-bundled 64-bit JDK 8 for a diagnostic-only Pico build. ' +
            'This artifact must not be treated as the reproducible JDK 11 release build.'
        )
    } else {
        throw "Unity 2022 Android build requires 64-bit JDK 11; found:`n$JavaVersion"
    }
}

$SourceRoot = [System.IO.Path]::GetFullPath($SourceRoot)
$OutputApk = [System.IO.Path]::GetFullPath($OutputApk)
if (-not (Test-Path -LiteralPath $SourceRoot)) {
    $SourceParent = Split-Path $SourceRoot -Parent
    New-Item -ItemType Directory -Path $SourceParent -Force | Out-Null
    Invoke-Checked git clone --branch v1.1.1 --depth 1 $Repository $SourceRoot
}
if (-not (Test-Path -LiteralPath (Join-Path $SourceRoot '.git') -PathType Container)) {
    throw "SourceRoot is not an XRoboToolkit Git checkout: $SourceRoot"
}

$Head = (& git -C $SourceRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $Head -ne $ExpectedCommit) {
    throw "Expected XRoboToolkit v1.1.1 commit $ExpectedCommit; found $Head"
}

$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& git -C $SourceRoot apply --reverse --check $PatchPath 2>$null
$AlreadyPatched = $LASTEXITCODE -eq 0
$ErrorActionPreference = $PreviousErrorActionPreference
if (-not $AlreadyPatched) {
    Invoke-Checked git -C $SourceRoot apply --check $PatchPath
    Invoke-Checked git -C $SourceRoot apply $PatchPath
}

$OutputDirectory = Split-Path $OutputApk -Parent
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$BuildLog = "$OutputApk.log"
$QuotedSourceRoot = '"' + $SourceRoot + '"'
$QuotedOutputApk = '"outputPath=' + $OutputApk + '"'
$QuotedSdkRoot = '"sdkRoot=' + $SdkRoot + '"'
$QuotedNdkRoot = '"ndkRoot=' + $NdkRoot + '"'
$QuotedJdkRoot = '"jdkRoot=' + $JdkRoot + '"'
$QuotedBuildLog = '"' + $BuildLog + '"'
$UnityArguments = @(
    '-batchmode',
    '-nographics',
    '-quit',
    '-buildTarget', 'Android',
    '-projectPath', $QuotedSourceRoot,
    '-executeMethod', 'HardenedBuild.BuildAndroid',
    $QuotedOutputApk,
    $QuotedSdkRoot,
    $QuotedNdkRoot,
    $QuotedJdkRoot,
    '-logFile', $QuotedBuildLog
)

$UnityProcess = Start-Process `
    -FilePath $UnityEditor `
    -ArgumentList $UnityArguments `
    -WindowStyle Hidden `
    -PassThru
$UnityProcess.WaitForExit()
$UnityExitCode = $UnityProcess.ExitCode
if ($UnityExitCode -ne 0 -or -not (Test-Path -LiteralPath $OutputApk -PathType Leaf)) {
    $Tail = if (Test-Path -LiteralPath $BuildLog) {
        (Get-Content -LiteralPath $BuildLog -Tail 40) -join [Environment]::NewLine
    } else {
        'Unity build log was not created.'
    }
    throw "XRoboToolkit hardened APK build failed.`n$Tail"
}

$ApkHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $OutputApk).Hash.ToLowerInvariant()
Write-Host "[PASS] Built $OutputApk"
Write-Host "[PASS] SHA256 $ApkHash"

if ($InstallToConnectedPico) {
    if ([string]::IsNullOrWhiteSpace($AdbPath)) {
        $AdbCommand = Get-Command adb -ErrorAction SilentlyContinue
        if ($AdbCommand) {
            $AdbPath = $AdbCommand.Source
        } else {
            $AdbPath = Join-Path $AndroidPlayer 'SDK\platform-tools\adb.exe'
        }
    }
    if (-not (Test-Path -LiteralPath $AdbPath -PathType Leaf)) {
        throw "adb not found; pass -AdbPath explicitly"
    }

    $DeviceLines = @(
        & $AdbPath devices |
            Select-Object -Skip 1 |
            Where-Object { $_ -match "`tdevice$" }
    )
    if ($DeviceLines.Count -ne 1) {
        throw "Expected exactly one authorized ADB device; found $($DeviceLines.Count)"
    }

    Invoke-Checked $AdbPath install -g -r $OutputApk
    Invoke-Checked $AdbPath shell am force-stop com.xrobotoolkit.client
    Invoke-Checked $AdbPath shell am start -n `
        'com.xrobotoolkit.client.hardened/com.unity3d.player.UnityPlayerActivity'
    $HardenedPid = ''
    foreach ($Attempt in 1..20) {
        $PidOutput = @(& $AdbPath shell pidof com.xrobotoolkit.client.hardened)
        if ($LASTEXITCODE -eq 0) {
            $HardenedPid = ($PidOutput -join '').Trim()
            if (-not [string]::IsNullOrWhiteSpace($HardenedPid)) {
                break
            }
        }
        Start-Sleep -Milliseconds 250
    }
    if ([string]::IsNullOrWhiteSpace($HardenedPid)) {
        throw 'Hardened XRoboToolkit installed but did not remain running'
    }
    Write-Host "[PASS] Installed and opened hardened PICO client (pid $HardenedPid)"
    Write-Host '[NEXT] In headset: set PC IP; enable Head, Controller, Full body, Send.'
}
