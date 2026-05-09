@echo off
title Real Time Dynamic Pricing Engine

echo =======================================================
echo Starting Real-Time Dynamic Pricing Engine
echo =======================================================

echo.
echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Starting Streamlit dashboard...
"venv\Scripts\python.exe" -m streamlit run app.py

pause
