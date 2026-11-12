@echo off
cd /d "%~dp0"
where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    py desktop_app.py
) else (
    python desktop_app.py
)
