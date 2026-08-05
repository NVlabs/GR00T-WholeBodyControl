@echo off
setlocal enabledelayedexpansion

echo Setting up Qt environment for Windows...

set QT_MSVC2019_64=C:\Qt6.0\6.6.2\msvc2019_64
set QT6_TOOLS=C:\Qt6.0\Tools

set PATH=%QT_MSVC2019_64%\bin;%PATH%
set PATH=%QT_MSVC2019_64%\include;%PATH%
set PATH=%QT6_TOOLS%\QtCreator\bin;%PATH%
set PATH=%QT6_TOOLS%\QtCreator\bin\jom;%PATH%
set PATH=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build;%PATH%
set PATH=%QT6_TOOLS%\CMake_64\bin;%PATH%

set SCRIPT_DIR=%~dp0
cd %SCRIPT_DIR%
echo Working directory: %SCRIPT_DIR%

set BUILD_DIR=%SCRIPT_DIR%build

if not exist "%BUILD_DIR%" (
    echo Creating build directory...
    mkdir "%BUILD_DIR%"
)

if "%1"=="--clean" (
    echo Cleaning build directory...
    if exist "%BUILD_DIR%" (
        rmdir /S /Q "%BUILD_DIR%"
        mkdir "%BUILD_DIR%"
    )
)

rem Configure application arch (eg. x86 or amd64)
call vcvarsall.bat amd64

echo Starting build process...

where cl.exe > %TEMP%\output.txt
set /p MSVC_COMPILER=<%TEMP%\output.txt
echo MSVC compiler path: %MSVC_COMPILER%

if not exist "%BUILD_DIR%\CMakeCache.txt" (
    echo Configuring project with CMake...
    cmake.exe -S "%SCRIPT_DIR%" -B "%BUILD_DIR%" -DCMAKE_GENERATOR:STRING=Ninja -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo -DCMAKE_PREFIX_PATH:PATH=%QT_MSVC2019_64% "-DCMAKE_C_COMPILER:FILEPATH=%MSVC_COMPILER%" "-DCMAKE_CXX_COMPILER:FILEPATH=%MSVC_COMPILER%"
)

echo Building project...
cmake.exe --build "%BUILD_DIR%" --config RelWithDebInfo

if %ERRORLEVEL% EQU 0 (
    echo Build successful!

    if exist "%BUILD_DIR%\RobotDemoQt.exe" (
        echo Copying executable to Redistributable directory...
        if not exist "%SCRIPT_DIR%..\..\..\Redistributable\win" (
            mkdir "%SCRIPT_DIR%..\..\..\Redistributable\win"
        )
        copy "%BUILD_DIR%\RobotDemoQt.exe" "%SCRIPT_DIR%..\..\..\Redistributable\win\"
        echo Executable copied to %SCRIPT_DIR%..\..\..\Redistributable\win\RobotDemoQt.exe
    ) else (
        echo Warning: Executable not found at expected location: %BUILD_DIR%\RobotDemoQt.exe
        echo Searching for the executable...
        dir /S /B "%BUILD_DIR%\*.exe"
    )
) else (
    echo Build failed!
    exit /b 1
)

echo Build process completed.
exit /b 0
