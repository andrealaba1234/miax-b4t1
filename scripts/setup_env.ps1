param(
    [string]$VenvPath = ".venv",
    [string]$KernelName = "miax-b4t1",
    [string]$KernelDisplayName = "Python (miax-b4t1)"
)

$ErrorActionPreference = "Stop"

function Get-PythonLauncher {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $installed = (& py -0p 2>$null) -join "`n"
        foreach ($minor in @("3.10", "3.11", "3.12", "3.13")) {
            if ($installed -match [regex]::Escape($minor)) {
                return @{
                    Exe  = "py"
                    Args = @("-$minor")
                }
            }
        }
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @{
            Exe  = "python"
            Args = @()
        }
    }

    throw "No se encontro un interprete de Python instalado."
}

$py = Get-PythonLauncher
$version = (& $py.Exe @($py.Args + @("-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"))).Trim()

Write-Host "Usando Python $version via $($py.Exe) $($py.Args -join ' ')"
if (-not $version.StartsWith("3.10")) {
    Write-Warning "El repositorio esta verificado con Python 3.10. Continuo con $version porque no hay 3.10 disponible."
}

if (-not (Test-Path $VenvPath)) {
    Write-Host "Creando entorno virtual en $VenvPath ..."
    & $py.Exe @($py.Args + @("-m", "venv", $VenvPath))
} else {
    Write-Host "Reutilizando entorno existente en $VenvPath ..."
}

$venvPython = Join-Path $VenvPath "Scripts\\python.exe"
$venvPip = Join-Path $VenvPath "Scripts\\pip.exe"

if (-not (Test-Path $venvPython)) {
    throw "No se encontro $venvPython tras crear el entorno."
}

Write-Host "Actualizando pip ..."
& $venvPython -m pip install --upgrade pip

Write-Host "Instalando dependencias de requirements.txt ..."
& $venvPip install -r requirements.txt

$zipPath = "data\\application_train.zip"
$csvPath = "data\\application_train.csv"
if ((Test-Path $zipPath) -and (-not (Test-Path $csvPath))) {
    Write-Host "Descomprimiendo dataset en data\\ ..."
    Expand-Archive -Path $zipPath -DestinationPath "data" -Force
} elseif (Test-Path $csvPath) {
    Write-Host "Dataset ya descomprimido: $csvPath"
}

Write-Host "Registrando kernel de Jupyter ..."
& $venvPython -m ipykernel install --user --name $KernelName --display-name $KernelDisplayName

Write-Host ""
Write-Host "Entorno listo."
Write-Host "Activacion: .\\$VenvPath\\Scripts\\Activate.ps1"
Write-Host "Kernel Jupyter: $KernelDisplayName"
