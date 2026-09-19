@echo off
title Flytris arcade preview
cd /d "%~dp0"
echo Serving the 3D arcade at http://127.0.0.1:8765/  (close this window to stop)
start "" "http://127.0.0.1:8765/index.html?n=1500"
python -m http.server 8765 --bind 127.0.0.1
pause >nul
