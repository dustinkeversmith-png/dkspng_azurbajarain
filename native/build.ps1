$ErrorActionPreference = "Stop"

$Native = $PSScriptRoot
$Root = Split-Path $Native -Parent
$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
if ($env:PYTHON) {
    $Python = $env:PYTHON
} elseif (Test-Path $venvPython) {
    $Python = $venvPython
} else {
    $Python = (Get-Command python).Source
}
$Gxx = "C:\msys64\mingw64\bin\g++.exe"
$DllTool = "C:\msys64\mingw64\bin\dlltool.exe"

if (-not (Test-Path $Gxx)) {
    throw "mingw-w64 g++ was not found at $Gxx"
}
if (-not (Test-Path $DllTool)) {
    throw "dlltool was not found at $DllTool"
}

& $Python -m pip install --disable-pip-version-check pybind11

$pybindInclude = (& $Python -c "import pybind11; print(pybind11.get_include())").Trim()
$pythonInclude = (& $Python -c "import sysconfig; print(sysconfig.get_path('include'))").Trim()
$dllInfo = (& $Python -c @"
import sys, sysconfig
from pathlib import Path
name = sysconfig.get_config_var('LDLIBRARY') or f'python{sys.version_info.major}{sys.version_info.minor}.dll'
base = Path(sys.base_prefix)
hits = [base / name, Path(sys.executable).resolve().parent / name]
found = next((path for path in hits if path.exists()), None)
if found is None:
    raise SystemExit(f'{name} was not found for {sys.executable}')
print(found)
print(found.name)
"@).Trim() -split '\r?\n'
$dll = $dllInfo[0].Trim()
$dllName = $dllInfo[1].Trim()
$importLib = Join-Path $Native ("lib" + [IO.Path]::GetFileNameWithoutExtension($dllName) + ".a")
$def = Join-Path $Native ([IO.Path]::GetFileNameWithoutExtension($dllName) + ".def")

if (-not (Test-Path $importLib)) {
    & $Python -c @"
import struct
from pathlib import Path
data = Path(r'$dll').read_bytes()

def u16(offset):
    return struct.unpack_from('<H', data, offset)[0]

def u32(offset):
    return struct.unpack_from('<I', data, offset)[0]

pe = u32(0x3C)
coff = pe + 4
sections = u16(coff + 2)
optional_size = u16(coff + 16)
optional = coff + 20
if u16(optional) != 0x20B:
    raise SystemExit('expected a PE32+ Python DLL')
export_rva = u32(optional + 112)
section_table = optional + optional_size
ranges = []
for index in range(sections):
    base = section_table + index * 40
    virtual_size = u32(base + 8)
    virtual_address = u32(base + 12)
    raw_size = u32(base + 16)
    raw_pointer = u32(base + 20)
    ranges.append((virtual_address, max(virtual_size, raw_size), raw_pointer))

def rva_to_offset(rva):
    for virtual_address, size, raw_pointer in ranges:
        if virtual_address <= rva < virtual_address + size:
            return raw_pointer + (rva - virtual_address)
    raise SystemExit(f'RVA {rva:#x} is not in any section')

export_dir = rva_to_offset(export_rva)
name_count = u32(export_dir + 24)
names_rva = u32(export_dir + 32)
names = rva_to_offset(names_rva)
lines = ['EXPORTS']
for index in range(name_count):
    name_rva = u32(names + index * 4)
    start = rva_to_offset(name_rva)
    end = data.index(b'\0', start)
    lines.append(data[start:end].decode('ascii'))
Path(r'$def').write_text('\n'.join(lines) + '\n', encoding='ascii')
"@
    & $DllTool -d $def -l $importLib -D $dllName
}

$tools = Join-Path $Root "tools"
New-Item -ItemType Directory -Force -Path $tools | Out-Null
function Build-Extension([string]$SourceName, [string]$OutputName) {
    $output = Join-Path $tools $OutputName
    & $Gxx -O3 -Wall -Wextra -shared -std=c++17 -DNDEBUG -DNOMINMAX -D_hypot=hypot `
        "-I$pybindInclude" "-I$pythonInclude" `
        (Join-Path $Native $SourceName) `
        $importLib `
        -static `
        -o $output
    Write-Host "Built $output"
}

Build-Extension "processor.cpp" "native_engine.pyd"
Build-Extension "tensor_engine.cpp" "tensor_engine.pyd"
