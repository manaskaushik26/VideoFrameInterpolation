# Lint RTL with Verilator.
#
#   .\sim\run_lint.ps1                                   # every module in rtl\
#   .\sim\run_lint.ps1 -Modules line_buffer_3x3
#
# Two Windows traps this exists to work around, both of which fail *silently*:
#
#   1. `verilator` in oss-cad-suite\bin is an extensionless Perl script, not an
#      executable. Invoking it from PowerShell does not run the linter -- Windows
#      pops a "how do you want to open this file?" dialog and the shell reports
#      exit 0, which reads exactly like a clean lint. Always call
#      verilator_bin.exe.
#
#   2. The suite's environment.ps1 sets VERILATOR_ROOT to a Unix path
#      (/yosyshq/share/verilator) that does not resolve on Windows, so
#      verilator_bin.exe then cannot find its built-in waiver files. It is
#      overridden below.
#
# Exits non-zero if any module has warnings, so this is safe to gate on.

param(
    [string[]] $Modules,
    [string]   $RtlDir   = "rtl",
    [string]   $OssRoot  = "$env:USERPROFILE\Downloads\oss-cad-suite"
)

$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo

$verilator = Join-Path $OssRoot "bin\verilator_bin.exe"
if (-not (Test-Path $verilator)) { Write-Error "verilator_bin.exe not found at $verilator"; exit 2 }
$env:VERILATOR_ROOT = Join-Path $OssRoot "share\verilator"
if (-not (Test-Path (Join-Path $env:VERILATOR_ROOT "include\verilated_std_waiver.vlt"))) {
    Write-Error "VERILATOR_ROOT looks wrong: $env:VERILATOR_ROOT"; exit 2
}

# Default to every .sv under rtl\, named by file stem.
if (-not $Modules) {
    $Modules = Get-ChildItem -Recurse -Path $RtlDir -Filter *.sv | ForEach-Object { $_.BaseName }
}

$failed = @()
foreach ($m in $Modules) {
    $src = Get-ChildItem -Recurse -Path $RtlDir -Filter "$m.sv" | Select-Object -First 1
    if (-not $src) { Write-Output "$m : SOURCE NOT FOUND"; $failed += $m; continue }

    # -y on every rtl subdirectory so hierarchical modules resolve their
    # submodules by filename. Requires file stem == module name, which is the
    # convention here.
    # @() on both sides: Get-ChildItem returns a bare object rather than an array
    # when it matches exactly one directory, and + then fails on it.
    $inc = @()
    foreach ($d in @(Get-ChildItem -Recurse -Directory -Path $RtlDir) + @(Get-Item $RtlDir)) {
        $inc += @("-y", $d.FullName)
    }

    # -Wno-PINCONNECTEMPTY allows `.pin()` -- an explicitly unconnected port --
    # while PINMISSING (a port omitted entirely) stays an error. That is the
    # distinction that matters here: deliberate non-use is fine and is always
    # commented; a silently dropped signal is the defect class that produced the
    # constant-zero design, and it still fails the lint.
    $out = & $verilator --lint-only -Wall -Wno-DECLFILENAME -Wno-PINCONNECTEMPTY `
                        @inc --top-module $m $src.FullName 2>&1
    if ($LASTEXITCODE -eq 0) {
        "{0,-24} clean" -f $m
    } else {
        "{0,-24} WARNINGS" -f $m
        $out | Where-Object { $_ -match '%(Warning|Error)' } | ForEach-Object { "    $_" }
        $failed += $m
    }
}

if ($failed.Count) { Write-Output ""; Write-Output "FAILED: $($failed -join ', ')"; exit 1 }
Write-Output ""; Write-Output "all clean"
exit 0
