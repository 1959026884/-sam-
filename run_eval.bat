@echo off
setlocal
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "PYTHONUTF8=1"

set "PYTHONEXE=%SAM_PYTHON%"
if not defined PYTHONEXE if exist "%~dp0.venv\Scripts\python.exe" set "PYTHONEXE=%~dp0.venv\Scripts\python.exe"
if not defined PYTHONEXE if exist "%USERPROFILE%\.conda\envs\chemprop2\python.exe" set "PYTHONEXE=%USERPROFILE%\.conda\envs\chemprop2\python.exe"
if not defined PYTHONEXE set "PYTHONEXE=python"

"%PYTHONEXE%" "%~dp0src\evaluate.py" %*
exit /b %ERRORLEVEL%
