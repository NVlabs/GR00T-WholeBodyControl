@echo off
setlocal

echo Setting up environment for XRoboToolkit-PC-Service...

:: Define the base directory for the script execution
set "SCRIPT_ROOT=%CD%"

:: Define a temporary directory for cloning
set "TEMP_DIR=tmp"

:: Define source paths relative to the cloned repository root
set "XROBOTKIT_CLONED_REPO_PATH=%TEMP_DIR%\XRoboToolkit-PC-Service"
set "PXREAROBOTSDK_SOURCE_DIR=%XROBOTKIT_CLONED_REPO_PATH%\RoboticsService\PXREARobotSDK"
set "PXREAROBOTSDK_LIB_DIR=%XROBOTKIT_CLONED_REPO_PATH%\RoboticsService\SDK\win\64"

:: Define destination directories
set "LIB_DEST_DIR=%SCRIPT_ROOT%\lib"
set "INCLUDE_DEST_DIR=%SCRIPT_ROOT%\include"

:: Create destination directories
echo Creating destination directories...
mkdir "%LIB_DEST_DIR%" 2>NUL
if not exist "%LIB_DEST_DIR%" (
    echo Error: Failed to create lib directory. Exiting.
    exit /b 1
)

mkdir "%INCLUDE_DEST_DIR%" 2>NUL
if not exist "%INCLUDE_DEST_DIR%" (
    echo Error: Failed to create include directory. Exiting.
    exit /b 1
)

echo Destination directories created successfully.

:: --- Check for pybind11 and install if not found ---
echo.
echo Checking for pybind11...
pip show pybind11 >NUL 2>&1
if %errorlevel% neq 0 (
    echo Error: pybind11 not found. Please run `pip install pybind11` first. Exiting.
    exit /b 1
)

:: --- Set PYBIND11_DIR for CMake ---
echo.
echo Setting PYBIND11_DIR environment variable...
for /f "usebackq" %%i in (`python -c "import sys; print(sys.prefix)"`) do set PYTHON_PREFIX=%%i
if not defined PYTHON_PREFIX (
    echo Error: Could not determine Python installation prefix.
    echo Please ensure Python is correctly installed and in your PATH. Exiting.
    exit /b 1
)

set "PYBIND11_DIR=%PYTHON_PREFIX%\Lib\site-packages\pybind11\share\cmake\pybind11"
echo Attempting to set PYBIND11_DIR to: %PYBIND11_DIR%
set PYBIND11_DIR=%PYBIND11_DIR%
if not exist "%PYBIND11_DIR%\pybind11Config.cmake" (
    echo Warning: pybind11Config.cmake not found at expected PYBIND11_DIR: "%PYBIND11_DIR%"
    echo This might indicate a problem with the pybind11 installation or its path.
    :: Attempting to find another common path if the standard one doesn't work.
    for /d %%d in ("%PYTHON_PREFIX%\Lib\site-packages\pybind11\share\cmake\*") do (
        if exist "%%d\pybind11Config.cmake" (
            set "PYBIND11_DIR=%%d"
            echo Found pybind11Config.cmake in "%%d". Using this path.
            goto :pybind11_dir_found
        )
    )
    echo Critical Error: pybind11Config.cmake could not be found after pybind11 installation.
    echo Please check your pybind11 installation. Exiting.
    exit /b 1
)
:pybind11_dir_found
echo PYBIND11_DIR set to: %PYBIND11_DIR%

set "DLL_NAME=PXREARobotSDK.dll"
set "LIB_NAME=PXREARobotSDK.lib"

:: Create the temporary directory and navigate into it
echo Creating temporary directory: %TEMP_DIR%
mkdir %TEMP_DIR%
if not exist %TEMP_DIR% (
    echo Error: Failed to create temporary directory %TEMP_DIR%. Ingore.
)
cd %TEMP_DIR%
if %errorlevel% neq 0 (
    echo Error: Failed to navigate into %TEMP_DIR%. Exiting.
    exit /b 1
)

:: Clone the repository
echo Cloning XRoboToolkit-PC-Service repository...
git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git
if %errorlevel% neq 0 (
    echo Error: Git clone failed. Exiting.
    cd ..
    rmdir /s /q %TEMP_DIR%
    exit /b 1
)

:: Navigate back to the script's root directory to handle destinations
cd %SCRIPT_ROOT%
if %errorlevel% neq 0 (
    echo Error: Failed to navigate back to script root. Exiting.
    exit /b 1
)

:: --- Copy Header Files ---
echo.
echo Copying header files to %INCLUDE_DEST_DIR%...

:: Copy PXREARobotSDK.h
set "PXREAROBOTSDK_H_SRC=%PXREAROBOTSDK_SOURCE_DIR%\PXREARobotSDK.h"
echo Copying %PXREAROBOTSDK_H_SRC%
copy "%PXREAROBOTSDK_H_SRC%" "%INCLUDE_DEST_DIR%\"
if %errorlevel% neq 0 (
    echo Error: Failed to copy PXREARobotSDK.h. Exiting.
    goto :cleanup_and_exit
)

:: Create nlohmann subdirectory in include
set "NLOHMANN_INCLUDE_DEST_DIR=%INCLUDE_DEST_DIR%\nlohmann"
echo Ensuring '%NLOHMANN_INCLUDE_DEST_DIR%' directory exists...
mkdir %NLOHMANN_INCLUDE_DEST_DIR% 2>NUL
if %errorlevel% neq 0 (
    echo Error: Failed to create nlohmann include directory. Ignore.
)

:: Copy nlohmann/json.hpp
set "NLOHMANN_JSON_HPP_SRC=%PXREAROBOTSDK_SOURCE_DIR%\nlohmann\json.hpp"
echo Copying %NLOHMANN_JSON_HPP_SRC%
copy "%NLOHMANN_JSON_HPP_SRC%" "%NLOHMANN_INCLUDE_DEST_DIR%\"
if %errorlevel% neq 0 (
    echo Error: Failed to copy nlohmann/json.hpp. Exiting.
    goto :cleanup_and_exit
)

:: Copy nlohmann/json_fwd.hpp
set "NLOHMANN_JSON_FWD_HPP_SRC=%PXREAROBOTSDK_SOURCE_DIR%\nlohmann\json_fwd.hpp"
echo Copying %NLOHMANN_JSON_FWD_HPP_SRC%
copy "%NLOHMANN_JSON_FWD_HPP_SRC%" "%NLOHMANN_INCLUDE_DEST_DIR%\"
if %errorlevel% neq 0 (
    echo Error: Failed to copy nlohmann/json_fwd.hpp. Exiting.
    goto :cleanup_and_exit
)

echo Header files copied successfully.

:: --- Copy Pre-built PXREARobotSDK DLL and LIB ---
echo.
echo Checking for pre-built libraries in %PXREAROBOTSDK_LIB_DIR%
set "DLL_SOURCE_PATH=%PXREAROBOTSDK_LIB_DIR%\%DLL_NAME%"
set "LIB_SOURCE_PATH=%PXREAROBOTSDK_LIB_DIR%\%LIB_NAME%"

if not exist "%DLL_SOURCE_PATH%" (
    echo Error: Required DLL "%DLL_SOURCE_PATH%" not found.
    echo Please ensure the cloned repository contains the pre-built files.
    goto :cleanup_and_exit
)
if not exist "%LIB_SOURCE_PATH%" (
    echo Error: Required LIB "%LIB_SOURCE_PATH%" not found.
    echo Please ensure the cloned repository contains the pre-built files.
    goto :cleanup_and_exit
)

echo Copying %DLL_NAME% to %LIB_DEST_DIR%/
copy "%DLL_SOURCE_PATH%" "%LIB_DEST_DIR%\"
if %errorlevel% neq 0 (
    echo Error: Failed to copy %DLL_NAME%. Exiting.
    goto :cleanup_and_exit
)

echo Copying %LIB_NAME% to %LIB_DEST_DIR%/
copy "%LIB_SOURCE_PATH%" "%LIB_DEST_DIR%\"
if %errorlevel% neq 0 (
    echo Error: Failed to copy %LIB_NAME%. Exiting.
    goto :cleanup_and_exit
)

echo Libraries copied successfully.

:: Build and install the Python project
echo.
echo Building and installing the Python project...
python setup.py install
if %errorlevel% neq 0 (
    echo Error: Python setup.py install failed. Exiting.
    goto :cleanup_and_exit
)

:: Copy DLL to the installed package location
echo.
echo Copying DLL to the installed package location...
for /f "usebackq" %%i in (`python -c "import site; print(site.getsitepackages()[0])"`) do set SITE_PACKAGES=%%i
if not defined SITE_PACKAGES (
    echo Warning: Could not determine site-packages directory.
    echo DLL not copied to package location. You may need to do this manually.
    goto :cleanup_and_exit
)

:: Find the egg directory
set "FOUND_EGG="
for /d %%d in ("%SITE_PACKAGES%\Lib\site-packages\xrobotoolkit_sdk-*") do (
    set "FOUND_EGG=%%d"
    goto :egg_found
)
:egg_found

if not defined FOUND_EGG (
    echo Warning: Could not find xrobotoolkit_sdk egg directory in %SITE_PACKAGES%
    echo Looking in easy-install.pth...
    if exist "%SITE_PACKAGES%\easy-install.pth" (
        for /f "usebackq tokens=*" %%i in (`findstr /i "xrobotoolkit_sdk" "%SITE_PACKAGES%\easy-install.pth"`) do set "FOUND_EGG=%%i"
    )
)

if not defined FOUND_EGG (
    echo Warning: Could not find xrobotoolkit_sdk egg directory.
    echo DLL not copied to package location. You may need to do this manually.
) else (
    echo Found egg directory: %FOUND_EGG%
    echo Copying %DLL_NAME% to %FOUND_EGG%
    copy "%LIB_DEST_DIR%\%DLL_NAME%" "%FOUND_EGG%\"
    if %errorlevel% neq 0 (
        echo Warning: Failed to copy DLL to egg directory.
    ) else (
        echo DLL successfully copied to package location.
    )
)

echo Setup completed successfully!

:cleanup_and_exit
:: Remove the temporary directory
echo Cleaning up temporary directory: %TEMP_DIR%
rmdir /s /q "%SCRIPT_ROOT%\%TEMP_DIR%"
if %errorlevel% neq 0 (
    echo Warning: Failed to remove temporary directory "%SCRIPT_ROOT%\%TEMP_DIR%". Please remove it manually.
)

endlocal