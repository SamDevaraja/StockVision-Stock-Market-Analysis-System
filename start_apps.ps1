# StockVision Startup Script
Write-Host "Starting StockVision..." -ForegroundColor Green

# Check if backend is running
$backendRunning = $false
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5000/health" -TimeoutSec 2 -UseBasicParsing
    if ($response.StatusCode -eq 200) {
        Write-Host "[OK] Backend API is already running on http://localhost:5000" -ForegroundColor Green
        $backendRunning = $true
    }
} catch {
    Write-Host "[INFO] Starting Backend API..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\Users\Sam Devaraja\Desktop\StockVision'; python backend/app.py"
    Start-Sleep -Seconds 3
    $backendRunning = $true
}

# Start Streamlit
Write-Host "[INFO] Starting Streamlit Frontend..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\Users\Sam Devaraja\Desktop\StockVision'; python -m streamlit run frontend/app.py"

Write-Host "Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 8

# Open browser
Write-Host "[INFO] Opening browser..." -ForegroundColor Yellow
Start-Process "http://localhost:8501"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "StockVision is starting!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Backend API:  http://localhost:5000" -ForegroundColor White
Write-Host "Frontend UI:  http://localhost:8501" -ForegroundColor White
Write-Host ""
Write-Host "The browser should open automatically." -ForegroundColor Yellow
Write-Host "If not, manually open: http://localhost:8501" -ForegroundColor Yellow

