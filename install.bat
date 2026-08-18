@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title FireRedTTS3 Easy GUI - Fixed CUDA Installer

set "UV_VERSION=0.10.0"
set "UV_URL=https://github.com/astral-sh/uv/releases/download/%UV_VERSION%/uv-x86_64-pc-windows-msvc.zip"
set "UV_ZIP=%CD%\.runtime\temp\uv-%UV_VERSION%-windows-x64.zip"
set "UV_EXTRACT_DIR=%CD%\.runtime\temp\uv-%UV_VERSION%-extract"

set "PYTHON_VERSION=3.12.10"
set "TORCH_VERSION=2.8.0"
set "CUDA_RUNTIME=12.8"
set "FA_VERSION=2.8.3"
set "TRITON_VERSION=3.4.0.post21"
set "FA_WHEEL_NAME=flash_attn-2.8.3+cu128torch2.8.0cxx11abiTRUE-cp312-cp312-win_amd64.whl"
set "FA_WHEEL_URL=https://huggingface.co/Wildminder/AI-windows-whl/resolve/main/%FA_WHEEL_NAME%?download=true"

set "UV_DIR=%CD%\.runtime\uv"
set "UV_EXE=%UV_DIR%\uv.exe"
set "PY_EXE=%CD%\.venv\Scripts\python.exe"

set "UV_PYTHON_INSTALL_DIR=%CD%\.runtime\python"
set "UV_PROJECT_ENVIRONMENT=%CD%\.venv"
set "UV_CACHE_DIR=%CD%\.runtime\uv-cache"

set "HF_HOME=%CD%\.runtime\hf-cache"
set "HUGGINGFACE_HUB_CACHE=%HF_HOME%\hub"
set "HF_HUB_CACHE=%HF_HOME%\hub"
set "HF_XET_CACHE=%HF_HOME%\xet"
set "TRANSFORMERS_CACHE=%HF_HOME%\transformers"

set "TMP=%CD%\.runtime\temp"
set "TEMP=%CD%\.runtime\temp"
set "GRADIO_TEMP_DIR=%CD%\.runtime\temp"

set "UV_NO_CACHE=1"
set "UV_LINK_MODE=copy"
set "PIP_NO_CACHE_DIR=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "TORCH_EXTENSIONS_DIR=%CD%\.runtime\torch-extensions"

for %%D in (
  ".runtime"
  ".runtime\temp"
  "%UV_DIR%"
  "models"
  "outputs"
  "voices"
  "config"
  "accelerator_wheels"
) do if not exist "%%~D" mkdir "%%~D"

where nvidia-smi >nul 2>nul
if errorlevel 1 (
  echo [ERROR] NVIDIA driver / nvidia-smi was not detected.
  goto :fail
)

echo [1/7] NVIDIA GPU detected:
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader


rem Remove obsolete experimental C++ / GGUF files from earlier patches.
if exist "fireredtts3_local\cpp_backend.py" del /q "fireredtts3_local\cpp_backend.py" >nul 2>&1
if exist "tools\export_firered_cpp_bundle.py" del /q "tools\export_firered_cpp_bundle.py" >nul 2>&1
if exist "tools\build_firered_cpp_runtime.bat" del /q "tools\build_firered_cpp_runtime.bat" >nul 2>&1
if exist "cpp_runtime" rmdir /s /q "cpp_runtime" >nul 2>&1
if exist ".runtime\firered-cpp" rmdir /s /q ".runtime\firered-cpp" >nul 2>&1
if exist "models\cpp-bundles" rmdir /s /q "models\cpp-bundles" >nul 2>&1

rem ============================================================
rem 2/7 - Project-local uv
rem ============================================================
if exist "%UV_EXE%" (
  echo [2/7] Project-local uv already present. Skipping.
) else (
  echo [2/7] Downloading project-local uv %UV_VERSION%...
  where curl.exe >nul 2>&1 || (
    echo [ERROR] curl.exe is required to bootstrap uv.
    goto :fail
  )
  where tar.exe >nul 2>&1 || (
    echo [ERROR] tar.exe is required to bootstrap uv.
    goto :fail
  )
  if exist "%UV_ZIP%" del /q "%UV_ZIP%" >nul 2>&1
  if exist "%UV_EXTRACT_DIR%" rmdir /s /q "%UV_EXTRACT_DIR%" >nul 2>&1
  mkdir "%UV_EXTRACT_DIR%" >nul 2>&1
  curl.exe -L --fail --retry 3 --retry-delay 2 -o "%UV_ZIP%" "%UV_URL%"
  if errorlevel 1 goto :fail
  tar.exe -xf "%UV_ZIP%" -C "%UV_EXTRACT_DIR%"
  if errorlevel 1 goto :fail
  for /r "%UV_EXTRACT_DIR%" %%F in (uv.exe) do (
    if exist "%%~fF" if not exist "%UV_EXE%" copy /y "%%~fF" "%UV_EXE%" >nul
  )
  if not exist "%UV_EXE%" (
    echo [ERROR] uv.exe was not found inside the downloaded archive.
    goto :fail
  )
  del /q "%UV_ZIP%" >nul 2>&1
  rmdir /s /q "%UV_EXTRACT_DIR%" >nul 2>&1
)

rem ============================================================
rem 3/7 - Project-local Python
rem Never touches the global PATH, registry, or user .local\bin.
rem ============================================================
set "CURRENT_PYTHON="
if exist "%PY_EXE%" (
  for /f "delims=" %%V in ('"%PY_EXE%" -c "import sys; print('.'.join(map(str,sys.version_info[:3])))" 2^>nul') do set "CURRENT_PYTHON=%%V"
)

if /I "%CURRENT_PYTHON%"=="%PYTHON_VERSION%" (
  echo [3/7] Project environment already uses Python %PYTHON_VERSION%. Skipping.
) else (
  set "LOCAL_PY_READY="
  "%UV_EXE%" python find %PYTHON_VERSION% >nul 2>&1
  if not errorlevel 1 set "LOCAL_PY_READY=1"
  if defined LOCAL_PY_READY (
    echo [3/7] Project-local Python %PYTHON_VERSION% already available. Skipping download.
  ) else (
    echo [3/7] Installing project-local Python %PYTHON_VERSION%...
    "%UV_EXE%" python install %PYTHON_VERSION% --no-cache --no-bin --no-registry
    if errorlevel 1 goto :fail
  )
)

rem ============================================================
rem 4/7 - Frozen main environment
rem One sync only when versions do not already match.
rem Torch is resolved here from the CUDA 12.8 index, not reinstalled later.
rem ============================================================
set "MAIN_ENV_READY="
if exist "%PY_EXE%" (
  "%PY_EXE%" -c "import importlib.metadata as m, torch; assert torch.__version__=='2.8.0+cu128'; assert m.version('torchaudio').startswith('2.8.0'); assert m.version('torchcodec')=='0.7.0'; assert m.version('transformers')=='5.6.2'; assert m.version('einops')=='0.8.2'; assert m.version('faster-whisper')=='1.2.1'; assert m.version('gradio')=='6.24.0'; assert m.version('huggingface-hub')=='1.27.0'; assert m.version('hf-xet')=='1.6.0'; assert m.version('wetext')=='0.1.6'; assert m.version('peft')=='0.19.1'; assert m.version('tensorboard')=='2.20.0'; assert m.version('setuptools')=='78.1.0'; assert torch.cuda.is_available(); assert torch.version.cuda=='12.8'" >nul 2>&1
  if not errorlevel 1 set "MAIN_ENV_READY=1"
)

if defined MAIN_ENV_READY (
  echo [4/7] Frozen FireRedTTS3 environment already satisfied. Skipping uv sync.
) else (
  echo [4/7] Synchronizing frozen FireRedTTS3 environment...
  "%UV_EXE%" sync --no-cache --python %PYTHON_VERSION%
  if errorlevel 1 goto :fail
)

if not exist "%PY_EXE%" (
  echo [ERROR] Project environment was not created.
  goto :fail
)

rem ============================================================
rem 5/7 - FlashAttention Windows
rem Install the prebuilt wheel WITHOUT dependencies so it can never replace
rem the frozen Torch/CUDA runtime.
rem ============================================================
set "FLASH_READY="
"%PY_EXE%" -c "import importlib.metadata as m, torch; assert torch.__version__=='2.8.0+cu128'; assert m.version('flash-attn').split('+')[0]=='%FA_VERSION%'; import flash_attn; from flash_attn import flash_attn_func" >nul 2>&1
if not errorlevel 1 set "FLASH_READY=1"

if defined FLASH_READY (
  echo [5/7] FlashAttention %FA_VERSION% already installed and importable. Skipping.
) else (
  echo [5/7] Installing fixed FlashAttention Windows wheel...
  call :install_flash
  if errorlevel 1 goto :fail
)

rem ============================================================
rem 6/7 - Triton Windows for torch.compile
set "TRITON_READY="
"%PY_EXE%" -c "import importlib.metadata as m; assert m.version('triton-windows')=='%TRITON_VERSION%'; import triton" >nul 2>&1
if not errorlevel 1 set "TRITON_READY=1"
if defined TRITON_READY (
  echo [6/8] triton-windows %TRITON_VERSION% already installed. Skipping.
) else (
  echo [6/8] Installing triton-windows %TRITON_VERSION% for torch.compile...
  "%UV_EXE%" pip install --python "%PY_EXE%" --no-cache --no-deps "triton-windows==%TRITON_VERSION%"
  if errorlevel 1 goto :fail
)

rem ============================================================
rem 7/8 - Runtime verification
rem ============================================================
echo [7/8] Verifying CUDA runtime...
"%PY_EXE%" -c "import torch, importlib.metadata as m; import flash_attn; from flash_attn import flash_attn_func; print('[python]', __import__('sys').version.split()[0]); print('[torch]', torch.__version__); print('[cuda]', torch.version.cuda); print('[flash-attn]', m.version('flash-attn')); print('[gpu]', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE'); raise SystemExit(0 if torch.cuda.is_available() and torch.__version__=='2.8.0+cu128' and torch.version.cuda=='12.8' else 1)"
if errorlevel 1 goto :fail

rem ============================================================
rem 8/8 - Complete
rem ============================================================
echo [8/8] Installation complete.
echo.
echo ============================================================
echo  FireRedTTS3 Easy GUI is ready
echo ============================================================
echo Python: %PYTHON_VERSION%
echo PyTorch: %TORCH_VERSION% + CUDA %CUDA_RUNTIME%
echo FlashAttention: %FA_VERSION%
echo.
echo Run:
echo   start.bat
echo ============================================================
echo.
pause
exit /b 0

:install_flash
for %%F in ("accelerator_wheels\flash_attn*.whl") do if exist "%%~fF" (
  echo [FlashAttention] Using local wheel: %%~nxF
  "%UV_EXE%" pip install --python "%PY_EXE%" --no-cache --no-deps --reinstall "%%~fF"
  exit /b %errorlevel%
)

"%UV_EXE%" pip install --python "%PY_EXE%" --no-cache --no-deps --reinstall "%FA_WHEEL_URL%"
exit /b %errorlevel%

:fail
if exist "%UV_CACHE_DIR%" rmdir /s /q "%UV_CACHE_DIR%" >nul 2>&1
echo.
echo INSTALLATION FAILED. Review the error above.
pause
exit /b 1
