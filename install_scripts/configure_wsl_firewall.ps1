#Requires -RunAsAdministrator

[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateNotNullOrEmpty()]
    [string]$PicoSubnet,
    [ValidateNotNullOrEmpty()]
    [string]$UnitreeSubnet = '192.168.123.0/24',
    [string]$WslVmCreatorId = '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function ConvertTo-PrivateIPv4NetworkCidr {
    param(
        [Parameter(Mandatory)]
        [string]$Value,
        [Parameter(Mandatory)]
        [string]$ParameterName
    )

    # Keep parsing stepwise for Windows PowerShell 5.1 compatibility. A single
    # ValidateScript expression that mutates and repeatedly reads a [ref]
    # TryParse target can hang during parameter binding on that runtime.
    $cidrMatch = [System.Text.RegularExpressions.Regex]::Match(
        $Value,
        '\A(?<a>0|[1-9][0-9]{0,2})\.(?<b>0|[1-9][0-9]{0,2})\.(?<c>0|[1-9][0-9]{0,2})\.(?<d>0|[1-9][0-9]{0,2})/(?<prefix>0|[1-9][0-9]?)\z',
        [System.Text.RegularExpressions.RegexOptions]::CultureInvariant
    )
    if (-not $cidrMatch.Success) {
        throw "$ParameterName must be a canonical dotted-decimal IPv4 CIDR, for example 192.168.1.0/24."
    }

    [int[]]$octets = @(
        [int]$cidrMatch.Groups['a'].Value
        [int]$cidrMatch.Groups['b'].Value
        [int]$cidrMatch.Groups['c'].Value
        [int]$cidrMatch.Groups['d'].Value
    )
    foreach ($octet in $octets) {
        if ($octet -gt 255) {
            throw "$ParameterName contains an IPv4 octet outside 0-255."
        }
    }

    [int]$prefixLength = [int]$cidrMatch.Groups['prefix'].Value
    if ($prefixLength -lt 1 -or $prefixLength -gt 32) {
        throw "$ParameterName prefix length must be between 1 and 32; /0 is forbidden."
    }

    [int]$minimumPrivatePrefix = 0
    if ($octets[0] -eq 10) {
        $minimumPrivatePrefix = 8
    }
    elseif ($octets[0] -eq 172 -and $octets[1] -ge 16 -and $octets[1] -le 31) {
        $minimumPrivatePrefix = 12
    }
    elseif ($octets[0] -eq 192 -and $octets[1] -eq 168) {
        $minimumPrivatePrefix = 16
    }
    else {
        throw "$ParameterName must be wholly inside an RFC1918 private IPv4 network."
    }
    if ($prefixLength -lt $minimumPrivatePrefix) {
        throw "$ParameterName prefix extends outside its RFC1918 private network."
    }

    [uint64]$addressValue = (
        ([uint64]$octets[0] * 16777216) +
        ([uint64]$octets[1] * 65536) +
        ([uint64]$octets[2] * 256) +
        [uint64]$octets[3]
    )
    [uint64]$hostMask = 0
    if ($prefixLength -lt 32) {
        $hostMask = [uint64]([System.Math]::Pow(2, 32 - $prefixLength) - 1)
    }
    if (($addressValue -band $hostMask) -ne 0) {
        throw "$ParameterName must use the network address; host bits are set in $Value."
    }

    return $Value
}

# Validate every remote network before making any firewall change.
$PicoSubnet = ConvertTo-PrivateIPv4NetworkCidr `
    -Value $PicoSubnet `
    -ParameterName 'PicoSubnet'
$UnitreeSubnet = ConvertTo-PrivateIPv4NetworkCidr `
    -Value $UnitreeSubnet `
    -ParameterName 'UnitreeSubnet'

function Add-ScopedHyperVRule {
    param(
        [Parameter(Mandatory)]
        [string]$Name,
        [Parameter(Mandatory)]
        [string]$DisplayName,
        [Parameter(Mandatory)]
        [ValidateSet('TCP', 'UDP')]
        [string]$Protocol,
        [Parameter(Mandatory)]
        [string]$LocalPorts,
        [Parameter(Mandatory)]
        [string]$RemoteAddresses
    )

    $existing = Get-NetFirewallHyperVRule `
        -PolicyStore PersistentStore `
        -Name $Name `
        -ErrorAction SilentlyContinue
    if ($null -ne $existing) {
        Set-NetFirewallHyperVRule `
            -InputObject $existing `
            -NewDisplayName $DisplayName `
            -Direction Inbound `
            -VMCreatorId $WslVmCreatorId `
            -Protocol $Protocol `
            -LocalPorts $LocalPorts `
            -RemoteAddresses $RemoteAddresses `
            -Action Allow `
            -Enabled True | Out-Null
        Write-Host "[OK] Updated: $DisplayName ($Protocol $LocalPorts from $RemoteAddresses)"
        return
    }

    # Write explicitly to the same persistent store queried above.
    New-NetFirewallHyperVRule `
        -PolicyStore PersistentStore `
        -Name $Name `
        -DisplayName $DisplayName `
        -Direction Inbound `
        -VMCreatorId $WslVmCreatorId `
        -Protocol $Protocol `
        -LocalPorts $LocalPorts `
        -RemoteAddresses $RemoteAddresses `
        -Action Allow `
        -Enabled True | Out-Null
    Write-Host "[OK] Added: $DisplayName ($Protocol $LocalPorts from $RemoteAddresses)"
}

Add-ScopedHyperVRule `
    -Name 'Codex-PICO-XRobo-TCP-63901' `
    -DisplayName 'Codex PICO XRobo TCP 63901' `
    -Protocol TCP `
    -LocalPorts '63901' `
    -RemoteAddresses $PicoSubnet

Add-ScopedHyperVRule `
    -Name 'Codex-PICO-XRobo-UDP-29888' `
    -DisplayName 'Codex PICO XRobo UDP 29888' `
    -Protocol UDP `
    -LocalPorts '29888' `
    -RemoteAddresses $PicoSubnet

Add-ScopedHyperVRule `
    -Name 'Codex-Unitree-DDS-Discovery' `
    -DisplayName 'Codex Unitree DDS discovery' `
    -Protocol UDP `
    -LocalPorts '7400-7401' `
    -RemoteAddresses $UnitreeSubnet

Add-ScopedHyperVRule `
    -Name 'Codex-Unitree-DDS-Unicast' `
    -DisplayName 'Codex Unitree DDS unicast' `
    -Protocol UDP `
    -LocalPorts '32768-65535' `
    -RemoteAddresses $UnitreeSubnet

Write-Host ''
Write-Host 'WSL Hyper-V firewall profile ready.'
Write-Host "PICO rules are scoped to the explicitly supplied subnet: $PicoSubnet"
Write-Host "Unitree rules are scoped to the validated subnet: $UnitreeSubnet"
Write-Host 'Existing unrelated firewall rules were not changed.'
