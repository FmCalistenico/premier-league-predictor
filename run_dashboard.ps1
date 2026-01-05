# Script para ejecutar el dashboard de Premier League Predictor
# Uso: .\run_dashboard.ps1

Write-Host "=== Premier League Predictor Dashboard ===" -ForegroundColor Cyan
Write-Host ""

# Verificar que estamos en el directorio correcto
if (-not (Test-Path "dashboards\bias_monitor.py")) {
    Write-Host "ERROR: Este script debe ejecutarse desde el directorio premier-league-predictor" -ForegroundColor Red
    exit 1
}

# Verificar venv activo
if ($env:VIRTUAL_ENV) {
    Write-Host "Virtual environment activo: $env:VIRTUAL_ENV" -ForegroundColor Green
} else {
    Write-Host "ADVERTENCIA: No hay virtual environment activo" -ForegroundColor Yellow
    Write-Host "Activalo con: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host ""
}

# Ejecutar dashboard
Write-Host "Iniciando dashboard Streamlit..." -ForegroundColor Cyan
Write-Host "URL: http://localhost:8501" -ForegroundColor Green
Write-Host ""
Write-Host "Presiona Ctrl+C para detener el servidor" -ForegroundColor Yellow
Write-Host ""

python -m streamlit run dashboards/bias_monitor.py
