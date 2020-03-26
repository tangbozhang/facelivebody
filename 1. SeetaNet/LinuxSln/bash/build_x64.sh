#!/usr/bin/env bash
g++ -m64 -shared -o libholiday.so -fPIC -O2 -fpermissive -std=c++11 \
-DSEETA_EXPORTS \
-I/opt/OpenBLAS/include -I/opt/protobuf/include -Iinclude -Isrc/include_inner -Isrc/proto \
src/proto/*.pb.cc src/*.cpp \
src/mem/*.cpp \
-L/opt/OpenBLAS/lib -lopenblas \
-L/opt/protobuf/lib -lprotobuf

