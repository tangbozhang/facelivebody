#!/usr/bin/env bash
export PROJECT_DIR=SeetaPassiveFaceAntiSpoofing
export OUT_DIR=../Cherry/Cherry/SEETAPassiveFASAndShakeHead/

echo copy into $OUT_DIR

if [ ! -d $OUT_DIR ]; then
    mkdir $OUT_DIR
fi

if [ ! -d $OUT_DIR/include ]; then
    mkdir $OUT_DIR/include
fi

cp release/*.a $OUT_DIR

cp $PROJECT_DIR/SEETAPassiveFASAndShakeHead.h $OUT_DIR/include
cp $PROJECT_DIR/SEETACPassiveFASAndShakeHead.h $OUT_DIR/include
cp $PROJECT_DIR/VIPL*.h $OUT_DIR/include

