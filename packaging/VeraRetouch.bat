@echo off
rem VeraRetouch 启动入口（控制台窗口即退出开关：关闭窗口 = 退出应用）
chcp 65001 >nul
title VeraRetouch
cd /d %~dp0
runtime\python.exe launcher.py
if errorlevel 1 pause
