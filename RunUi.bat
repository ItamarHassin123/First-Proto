@echo off
cd /d "%~dp0"

REM Create venv if missing
if not exist ".venv" (
    py -m venv .venv
)

REM Activate venv
call ".venv\Scripts\activate"

REM Install deps (first time)
pip install -q streamlit opencv-python torch torchvision pillow playsound3

REM Use the venv python explicitly (IMPORTANT)
set PYEXE=%cd%\.venv\Scripts\python.exe

REM Start Streamlit hidden, write logs to streamlit.log
powershell -NoProfile -WindowStyle Hidden -Command ^
"Start-Process -WindowStyle Hidden -FilePath '%PYEXE%' -ArgumentList '-m','streamlit','run','app.py','--server.headless=true','--server.port=8501' -RedirectStandardOutput"

REM Wait a moment
timeout /t 2 /nobreak >nul

REM Open browser
start "" "http://localhost:8501"

exit
