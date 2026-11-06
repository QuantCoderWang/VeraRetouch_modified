@echo off
rem VeraRetouch GPU acceleration enabler (one-time; requires NVIDIA GPU machine)
rem Content is ASCII-only on purpose: batch + UTF-8 only mixes reliably this way.
chcp 65001 >nul
title VeraRetouch GPU Setup
cd /d %~dp0
runtime\python.exe gpu_setup.py
pause
