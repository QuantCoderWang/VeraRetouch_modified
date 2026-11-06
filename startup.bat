@echo off
cd /d %~dp0
vera_venv\Scripts\python app.py --model-path ./checkpoints/VeraRetouch
pause
