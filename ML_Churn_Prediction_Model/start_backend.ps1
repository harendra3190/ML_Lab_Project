# Quick Start Script for Churn Prediction Project

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Customer Churn Prediction - Quick Start" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check if dataset exists
if (-not (Test-Path "Telco-Customer-Churn.csv")) {
    Write-Host "ERROR: Telco-Customer-Churn.csv not found!" -ForegroundColor Red
    exit 1
}

# Check if models exist
if (-not (Test-Path "model data/ls/churn_pipeline.pkl")) {
    Write-Host "Models not found. Training models first..." -ForegroundColor Yellow
    Write-Host "Step 1/2: Training ML models..." -ForegroundColor Green
    python train_model.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Model training failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host ""
} else {
    Write-Host "Models found. Skipping training..." -ForegroundColor Green
    Write-Host ""
}

Write-Host "Step 2/2: Starting Flask API server..." -ForegroundColor Green
Write-Host "API will be available at: http://localhost:5000" -ForegroundColor Cyan
Write-Host "Frontend: Open another terminal and run:" -ForegroundColor Yellow
Write-Host "  cd frontend" -ForegroundColor Yellow
Write-Host "  npm install" -ForegroundColor Yellow
Write-Host "  npm run dev" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""

python app.py

