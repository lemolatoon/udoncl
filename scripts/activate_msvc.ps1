[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

if ($env:VCToolsInstallDir -and (Get-Command cmake -ErrorAction SilentlyContinue) -and (Get-Command ninja -ErrorAction SilentlyContinue)) {
    Write-Host "UdonCL MSVC environment is already active."
    return
}

$programFilesX86 = [Environment]::GetEnvironmentVariable("ProgramFiles(x86)")
if (-not $programFilesX86) {
    throw "ProgramFiles(x86) is not defined."
}

$vswhere = Join-Path $programFilesX86 "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
    throw "Visual Studio Installer's vswhere.exe was not found: $vswhere"
}

$visualStudio = (& $vswhere -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
if (-not $visualStudio) {
    throw "Visual Studio with Desktop development with C++ was not found."
}

$vsDevCmd = Join-Path $visualStudio "Common7\Tools\VsDevCmd.bat"
if (-not (Test-Path -LiteralPath $vsDevCmd -PathType Leaf)) {
    throw "VsDevCmd.bat was not found: $vsDevCmd"
}

$environment = & $env:ComSpec /d /c "call `"$vsDevCmd`" -no_logo -arch=x64 -host_arch=x64 >nul && set"
if ($LASTEXITCODE -ne 0) {
    throw "VsDevCmd.bat failed with exit code $LASTEXITCODE."
}

foreach ($line in $environment) {
    $separator = $line.IndexOf("=")
    if ($separator -le 0) {
        continue
    }
    $name = $line.Substring(0, $separator)
    $value = $line.Substring($separator + 1)
    [Environment]::SetEnvironmentVariable($name, $value, "Process")
}

foreach ($command in @("cmake", "ninja", "cl")) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Visual Studio environment did not provide $command."
    }
}

Write-Host "UdonCL MSVC environment activated from $visualStudio"
