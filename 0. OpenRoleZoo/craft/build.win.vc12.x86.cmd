@echo off

set "BUILD_DIR=build.win.vc14.x86"
set "BUILD_TYPE=Release"
set "PLATFORM=x86"
set "PLATFORM_TARGET=x86"

set "WITH_SSL=ON"
set "SSL_HOME=D:\3rd\OpenSSL32"

set "INSTALL_DIR=D:\3rd\local"

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
