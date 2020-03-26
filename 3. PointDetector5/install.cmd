@echo off

if "%SEETASDK_HOME%" == "" goto no_seetasdk_home

set "InstallPrefix=%SEETASDK_HOME%\PDv500"

goto :default

:no_seetasdk_home
echo Can not detect SEETASDK_HOME. Please set it first.
goto :eof

:default

if not exist "%InstallPrefix%" mkdir "%InstallPrefix%"
echo.
echo Copy headers...
if not exist "%InstallPrefix%"\include mkdir "%InstallPrefix%"\include
xcopy /Y /S  "%~dp0"install\include\*.h "%InstallPrefix%"\include

echo Copy libraries...
if not exist "%InstallPrefix%"\lib mkdir "%InstallPrefix%"\lib
xcopy /Y /S  "%~dp0"install\lib\*.lib "%InstallPrefix%"\lib
xcopy /Y /S  "%~dp0"install\lib\*.dll "%InstallPrefix%"\lib

echo Install finished.

pause
