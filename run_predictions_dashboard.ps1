# Script para lanzar el Dashboard de Predicciones
# ================================================

Write-Host "Iniciando Dashboard de Predicciones..." -ForegroundColor Green

# Activar entorno virtual
Write-Host "Activando entorno virtual..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Verificar que streamlit está instalado
Write-Host "Verificando Streamlit..." -ForegroundColor Yellow
$streamlitCheck = python -m pip show streamlit 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "Streamlit no encontrado. Instalando..." -ForegroundColor Yellow
    python -m pip install streamlit plotly
}

# Lanzar dashboard
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Dashboard de Predicciones" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "El dashboard se abrira en tu navegador en:" -ForegroundColor Green
Write-Host "http://localhost:8501" -ForegroundColor Green
Write-Host ""
Write-Host "Presiona Ctrl+C para detener el servidor" -ForegroundColor Yellow
Write-Host ""

python -m streamlit run dashboard_predictions.py
