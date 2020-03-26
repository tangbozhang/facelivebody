#!/usr/bin/env bash
g++ -m32 -msse -shared -o libholiday.so -fPIC -O2 -fpermissive -std=c++11 \
-DSEETA_EXPORTS \
-I/opt/OpenBLAS32/include -I/opt/protobuf32/include -Iinclude -Isrc/include_inner -Isrc/proto \
src/proto/*.pb.cc src/*.cpp \
-L/opt/OpenBLAS32/lib -lopenblas \
-L/opt/protobuf32/lib -lprotobuf

