@echo off
setlocal
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "PYTHONUTF8=1"

set "PYTHONEXE=%SAM_GPU_PYTHON%"
if not defined PYTHONEXE set "PYTHONEXE=%USERPROFILE%\.conda\envs\chemprop2-gpu\python.exe"
if not exist "%PYTHONEXE%" (
  echo GPU Python environment not found: %PYTHONEXE%
  exit /b 2
)

"%PYTHONEXE%" "%~dp0src\cv_train.py" %*
exit /b %ERRORLEVEL%
