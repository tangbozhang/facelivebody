@echo off

set "BUILD_DIR=build.win.vc14.x64"
set "BUILD_TYPE=Release"
set "PLATFORM=amd64"
set "PLATFORM_TARGET=x64"

set "WITH_SSL=OFF"
set "SSL_HOME=D:\3rd\OpenSSL64"

set "INSTALL_DIR=E:\softwarebag\seeta"

call "%VS140COMNTOOLS%..\..\VC\vcvarsall.bat" %PLATFORM%

cd %~dp0

md "%BUILD_DIR%"

cd "%BUILD_DIR%"

md "%INSTALL_DIR%"

cmake "%~dp0.." ^
-G"NMake Makefiles JOM" ^
-DCMAKE_BUILD_TYPE="%BUILD_TYPE%" ^
-DPLATFORM="%PLATFORM_TARGET%" ^
-DORZ_WITH_OPENSSL="%WITH_SSL%" ^
-DOPENSSL_ROOT_DIR="%SSL_HOME%" ^
-DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"

jom -j16 install

exit /b

