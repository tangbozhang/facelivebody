#!/usr/bin/env bash

HOME=$(cd `dirname $0`; pwd)

echo ^=^= building x64 ^=^=
bash $HOME/build.linux.x64.sh

echo ^=^= building x64.debug ^=^=
bash $HOME/build.linux.x64.debug.sh

echo ^=^= building x86 ^=^=
bash $HOME/build.linux.x86.sh

echo ^=^= building x86.debug ^=^=
bash $HOME/build.linux.x86.debug.sh

