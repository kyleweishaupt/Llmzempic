@echo off
rem Creates .venv and installs Llmzempic with training support (Windows).
rem Override the PyTorch build with TORCH_INDEX_URL. Set WITH_UNSLOTH=1 to also install Unsloth.
setlocal
cd /d "%~dp0"

set "PY=python"
where py >nul 2>nul && set "PY=py -3"

%PY% -m venv .venv || goto :fail
set "VPY=.venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip || goto :fail

set "EXTRAS=train"
set "DEFAULT_INDEX=https://download.pytorch.org/whl/cpu"
where nvidia-smi >nul 2>nul && (
	set "EXTRAS=train,cuda"
	set "DEFAULT_INDEX=https://download.pytorch.org/whl/cu128"
)
if not defined TORCH_INDEX_URL set "TORCH_INDEX_URL=%DEFAULT_INDEX%"
echo Installing PyTorch from %TORCH_INDEX_URL%
"%VPY%" -m pip install torch --index-url %TORCH_INDEX_URL% || goto :fail
"%VPY%" -m pip install -e ".[%EXTRAS%]" || goto :fail
if "%WITH_UNSLOTH%"=="1" "%VPY%" -m pip install unsloth || goto :fail

echo.
echo Done. Start Llmzempic with start.cmd
exit /b 0

:fail
echo.
echo Setup failed - see the messages above.
exit /b 1
