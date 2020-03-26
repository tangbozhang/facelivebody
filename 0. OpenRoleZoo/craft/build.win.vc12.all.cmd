@echo off

echo ^=^= building x64 ^=^=
call %~dp0build.win.vc12.x64.cmd

echo ^=^= building x64.debug ^=^=
call %~dp0build.win.vc12.x64.debug.cmd

echo ^=^= building x86 ^=^=
call %~dp0build.win.vc12.x86.cmd

echo ^=^= building x86.debug ^=^=
call %~dp0build.win.vc12.x86.debug.cmd