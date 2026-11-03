# Run one testbench through the Vivado xsim flow.
#
#   .\sim\run_xsim.ps1 -Tb tb_line_buffer_3x3 -Sources rtl\conv\line_buffer_3x3.sv
#   .\sim\run_xsim.ps1 -Tb tb_line_buffer_3x3 -Sources rtl\conv\line_buffer_3x3.sv -Defines @{TB_IMG_W=224; TB_IMG_H=224}
#
# Vivado 2025.1 lives in a non-default location on this machine, so the path is
# set here rather than assumed to be on PATH. All xsim scratch output (xsim.dir,
# *.log, *.jou, *.pb, waveforms) is confined to build\ and is gitignored.

param(
    [Parameter(Mandatory=$true)][string]   $Tb,
    [Parameter(Mandatory=$true)][string[]] $Sources,
    [hashtable] $Defines = @{},
    [string]    $VivadoBin = "$env:USERPROFILE\Downloads\2025.1\Vivado\bin",
    [switch]    $KeepGoing
)

$ErrorActionPreference = "Continue"

if (-not (Test-Path $VivadoBin)) { Write-Error "Vivado bin not found: $VivadoBin"; exit 2 }
$env:PATH = "$VivadoBin;$env:PATH"

$repo  = Split-Path $PSScriptRoot -Parent
$build = Join-Path $repo "build"
New-Item -ItemType Directory -Force -Path $build | Out-Null

# Absolute source paths, since compilation happens from build\.
$tbFile  = Join-Path $repo "sim\$Tb.sv"
$absSrcs = @($tbFile) + ($Sources | ForEach-Object { Join-Path $repo $_ })
foreach ($s in $absSrcs) {
    if (-not (Test-Path $s)) { Write-Error "source not found: $s"; exit 2 }
}

# Options go through an -f file rather than the command line. PowerShell 5.1
# mangles native arguments containing '=', which silently turns "-d W=224" into
# a define plus a bogus filename, and the repo path contains a space besides.
$optsFile = Join-Path $build "$Tb.xvlog.f"
$opts = @("-sv")
foreach ($k in $Defines.Keys) { $opts += "-d $k=$($Defines[$k])" }
foreach ($s in $absSrcs)      { $opts += "`"$s`"" }
# ASCII, not utf8: PowerShell 5.1 writes a BOM for utf8, which corrupts the
# first line of the file and silently drops -sv, so the SystemVerilog sources
# then fail to parse as plain Verilog.
Set-Content -Path $optsFile -Value $opts -Encoding ascii

$snap = "${Tb}_snap"
Push-Location $build
try {
    $tag = if ($Defines.Count) { ($Defines.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ' ' } else { "defaults" }
    Write-Output "=== $Tb [$tag] ==="

    # No 2>&1 on these. PowerShell 5.1 wraps a native command's stderr in
    # NativeCommandError records, which makes any tool that merely prints a
    # warning look like a failure -- xelab did exactly that, and only for
    # parameter combinations that happened to emit one. Let the tools write
    # their own .log files instead and read those on failure.
    $log = xvlog -f $optsFile
    if ($LASTEXITCODE -ne 0) {
        Write-Output ($log | Select-String -Pattern 'ERROR' | Select-Object -First 15)
        if (Test-Path xvlog.log) { Get-Content xvlog.log | Select-String 'ERROR' | Select-Object -First 15 }
        Write-Output "XVLOG FAILED"; exit 1
    }

    # xelab's exit code is not trustworthy here. It builds the snapshot, then
    # tries to delete its scratch obj directory, and on this machine that
    # intermittently fails with "Access is denied" -- the repo lives under
    # OneDrive, whose sync client holds file handles. xelab then exits non-zero
    # despite having succeeded. Judge it on whether the snapshot actually
    # exists, and only report failure if it does not.
    # Remove any previous snapshot FIRST. Judging success by "the snapshot
    # exists" is only safe if a stale one cannot survive a failed elaboration --
    # otherwise a broken design silently re-runs the last good build and reports
    # PASS. That happened here: a port rename left vfi_top referencing pins that
    # no longer existed, xelab failed, and the previous snapshot ran instead.
    $snapDir = Join-Path "xsim.dir" $snap
    if (Test-Path $snapDir) { Remove-Item -Recurse -Force $snapDir -ErrorAction SilentlyContinue }

    $log = xelab -debug typical $Tb -s $snap
    $built = (Test-Path $snapDir) -or
             ($log | Select-String -Quiet 'Built simulation snapshot')
    if (-not $built) {
        Write-Output ($log | Select-String -Pattern 'ERROR' | Select-Object -First 15)
        if (Test-Path xelab.log) { Get-Content xelab.log | Select-String 'ERROR' | Select-Object -First 15 }
        Write-Output "XELAB FAILED"; exit 1
    }

    $out = xsim $snap --nolog -runall
    $out | Where-Object { $_ -notmatch '^\s*$' -and $_ -notmatch '^(INFO|WARNING)' }
    if ($LASTEXITCODE -ne 0 -and -not $KeepGoing) { exit 1 }
}
finally { Pop-Location }
