@echo off
setlocal

REM Function to check if a command exists
:check_command
where %1 >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo %1 not found.
    exit /b 1
)
exit /b 0

REM Check if Git is installed
echo Checking for Git...
call :check_command git
if %ERRORLEVEL% NEQ 0 (
    echo Git is not installed. Downloading and installing Git...
    echo Visit the following URL to download Git: https://git-scm.com/download/win
    pause
    exit /b 1
)

REM Check if Conda is installed
echo Checking for Conda...
call :check_command conda
if %ERRORLEVEL% NEQ 0 (
    echo Conda is not installed. Downloading and installing Miniconda...
    echo Visit the following URL to download Miniconda: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

REM Check if Python is installed
echo Checking for Python...
call :check_command python
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed. Installing Python...
    echo Visit the following URL to download Python: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Define repository URL and target directory
set REPO_URL=https://github.com/OpenBMB/MiniCPM-V.git
set TARGET_DIR=MiniCPM-V

REM Clone the repository
echo Cloning the repository...
git clone %REPO_URL%
if %ERRORLEVEL% NEQ 0 (
    echo Failed to clone the repository.
    pause
    exit /b 1
)

REM Navigate to the repository directory
cd %TARGET_DIR%

REM Create and activate Conda environment
echo Creating Conda environment...
conda create -n MiniCPM-V python=3.10 -y
if %ERRORLEVEL% NEQ 0 (
    echo Failed to create Conda environment.
    pause
    exit /b 1
)

REM Activate Conda environment
echo Activating Conda environment...
conda activate MiniCPM-V
if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate Conda environment.
    pause
    exit /b 1
)

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install dependencies.
    pause
    exit /b 1
)

echo Setup completed successfully.
pause

endlocal
