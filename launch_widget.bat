@echo off
REM AI ML VBUS Dashboard Launcher
REM Launches the current version of the widget

set CURRENT_VERSION=v1.0

echo Launching AI ML VBUS Dashboard %CURRENT_VERSION%...
start "" pythonw "%~dp0%CURRENT_VERSION%\ViewerDBX_GPU_Monitor.pyw"
