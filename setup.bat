@echo off
setlocal

:: Define the list of dependencies and their installation commands
set "requirements_file=requirements.txt"
set "python_executable=python" :: Adjust this if your Python executable has a different name or path

:: Check if Python is installed
echo Checking for Python...
where %python_executable% >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python first.
    exit /b 1
)

:: Check if pip is installed
echo Checking for pip...
%python_executable% -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Pip is not installed. Installing pip...
    %python_executable% -m ensurepip
    if %errorlevel% neq 0 (
        echo Failed to install pip. Please install pip manually.
        exit /b 1
    )
)

:: Check if the required packages are installed
echo Checking and installing dependencies from %requirements_file%...
for /f "tokens=*" %%i in (%requirements_file%) do (
    echo Checking if %%i is installed...
    %python_executable% -m pip show %%i >nul 2>&1
    if %errorlevel% neq 0 (
        echo %%i is not installed. Installing %%i...
        %python_executable% -m pip install %%i
        if %errorlevel% neq 0 (
            echo Failed to install %%i. Exiting...
            exit /b 1
        )
    ) else (
        echo %%i is already installed.
    )
)

:: Confirmation message
echo All dependencies are checked and installed.
pause
endlocal
