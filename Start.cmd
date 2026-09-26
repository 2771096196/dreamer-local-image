@echo off
chcp 65001 >nul
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
set "DREAMER_EXIT=%ERRORLEVEL%"
if not "%DREAMER_EXIT%"=="0" pause
exit /b %DREAMER_EXIT%
