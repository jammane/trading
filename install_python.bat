@echo off

REM Exit on error
setlocal EnableDelayedExpansion

REM Prompt for API keys
echo Yahoo Finance is used for market data and does not require an API key.

set /p ALPACA_API_KEY=Please enter your Alpaca API key: 
if "!ALPACA_API_KEY!"=="" (
    echo Error: Alpaca API key cannot be empty.
    exit /b 1
)

set /p ALPACA_SECRET_KEY=Please enter your Alpaca Secret key: 
if "!ALPACA_SECRET_KEY!"=="" (
    echo Error: Alpaca Secret key cannot be empty.
    exit /b 1
)

REM Set API keys for the current session
set ALPACA_API_KEY=!ALPACA_API_KEY!
set ALPACA_SECRET_KEY=!ALPACA_SECRET_KEY!

REM Persist API keys in system environment
echo Setting Alpaca API keys in system environment...
setx ALPACA_API_KEY "!ALPACA_API_KEY!"
if %ERRORLEVEL% neq 0 (
    echo Warning: Failed to set ALPACA_API_KEY in system environment.
)
setx ALPACA_SECRET_KEY "!ALPACA_SECRET_KEY!"
if %ERRORLEVEL% neq 0 (
    echo Warning: Failed to set ALPACA_SECRET_KEY in system environment.
)

REM Define Python installer details
set PYTHON_VERSION=3.12.0
set PYTHON_INSTALLER=python-%PYTHON_VERSION%-amd64.exe
set DOWNLOAD_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/%PYTHON_INSTALLER%

REM Download Python installer
echo Downloading Python %PYTHON_VERSION%...
powershell -Command "try { Invoke-WebRequest -Uri %DOWNLOAD_URL% -OutFile %PYTHON_INSTALLER% -ErrorAction Stop } catch { Write-Error 'Failed to download Python installer: $_'; exit 1 }"

REM Install Python silently, adding to PATH
echo Installing Python %PYTHON_VERSION%...
%PYTHON_INSTALLER% /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
if %ERRORLEVEL% neq 0 (
    echo Error: Python installation failed.
    exit /b 1
)

REM Verify Python installation
where py >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Python not found in PATH after installation.
    exit /b 1
)

REM Create a virtual environment
echo Creating virtual environment...
py -%PYTHON_VERSION% -m venv env
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to create virtual environment.
    exit /b 1
)

REM Activate the virtual environment
call env\Scripts\activate
if "%VIRTUAL_ENV%"=="" (
    echo Error: Failed to activate virtual environment.
    exit /b 1
)

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to upgrade pip.
    exit /b 1
)

REM Install required libraries
echo Installing required Python libraries...
pip install torch yfinance alpaca-py
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to install required libraries.
    exit /b 1
)

REM Verify library installations
for %%p in (torch yfinance alpaca-py) do (
    pip show %%p >nul 2>&1
    if !ERRORLEVEL! neq 0 (
        echo Error: Failed to install %%p.
        exit /b 1
    )
)

REM Update system PATH to include virtual environment Scripts directory
set VENV_SCRIPTS=%CD%\env\Scripts
echo Checking if virtual environment Scripts directory is in PATH...
echo %PATH% | findstr /C:"%VENV_SCRIPTS%" >nul
if %ERRORLEVEL% neq 0 (
    echo Updating system PATH...
    setx PATH "%PATH%;%VENV_SCRIPTS%"
    if %ERRORLEVEL% neq 0 (
        echo Warning: Failed to update system PATH. You may need to add %VENV_SCRIPTS% manually.
    ) else (
        echo System PATH updated successfully.
    )
) else (
    echo Virtual environment Scripts directory already in PATH.
)

echo Installation complete. Python %PYTHON_VERSION%, libraries, and Alpaca API keys configured.
echo Virtual environment created at %CD%\env.
echo To activate the virtual environment, run: call env\Scripts\activate
echo Note: PATH and API key changes via setx apply in new Command Prompt sessions.
