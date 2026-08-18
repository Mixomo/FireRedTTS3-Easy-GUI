@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title FireRedTTS3 Easy GUI

set "PY_EXE=%CD%\.venv\Scripts\python.exe"
set "HF_HOME=%CD%\.runtime\hf-cache"
set "HUGGINGFACE_HUB_CACHE=%HF_HOME%\hub"
set "HF_HUB_CACHE=%HF_HOME%\hub"
set "HF_XET_CACHE=%HF_HOME%\xet"
set "TRANSFORMERS_CACHE=%HF_HOME%\transformers"
set "TMP=%CD%\.runtime\temp"
set "TEMP=%CD%\.runtime\temp"
set "GRADIO_TEMP_DIR=%CD%\.runtime\temp"
set "PIP_NO_CACHE_DIR=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "TORCH_EXTENSIONS_DIR=%CD%\.runtime\torch-extensions"

if not exist "%PY_EXE%" (
  echo [ERROR] Project environment not installed. Run install.bat first.
  pause
  exit /b 1
)

for %%D in (".runtime\temp" "models" "outputs" "voices" "config" "accelerator_wheels") do if not exist "%%~D" mkdir "%%~D"

"%PY_EXE%" "%CD%\easy_gui.py"
if errorlevel 1 (
  echo.
  echo [ERROR] FireRedTTS3 Easy GUI exited with an error.
  pause
)
