@echo off
title Flytris final render (7,172 flies)
cd /d "%~dp0"
echo Rendering the finished video with music. Takes about 5 minutes.
python -X utf8 make_final.py
echo.
echo Done: media\flytris_arcade_3d_final.mp4
pause
