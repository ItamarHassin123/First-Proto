@echo off
cd /d "%~dp0"

REM ---- Create venv if missing ----
if not exist ".venv" (
    py -m venv .venv
)

REM ---- Activate venv ----
call .venv\Scripts\activate

REM ---- Install dependencies silently ----
pip install -q streamlit opencv-python torch torchvision pillow playsound3

REM ---- Launch Streamlit completely hidden via PowerShell ----
powershell -NoProfile -Command ^
"Start-Process python -ArgumentList '-m streamlit run app.py --server.headless=true --server.port=8501' -WindowStyle Hidden"

REM ---- Give server time to start ----
timeout /t 2 /nobreak >nul

REM ---- Open browser ----
start "" "http://localhost:8501"

exit
