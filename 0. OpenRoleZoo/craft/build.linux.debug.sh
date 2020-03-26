#!/usr/bin/env bash

export BUILD_DIR=build.linux.debug
export BUILD_TYPE=Debug

export WITH_SSL=ON

export INSTALL_DIR=/usr/local

HOME=$(cd `dirname $0`; pwd)

echo $HOME

cd $HOME

mkdir "$BUILD_DIR"

cd "$BUILD_DIR"

cmake "$HOME/.." \
-DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
-DORZ_WITH_OPENSSL="$WITH_SSL" \
-DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"

make -j16
make install


