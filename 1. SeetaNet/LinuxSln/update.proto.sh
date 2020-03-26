#!/usr/bin/env bash

export PROTO_DIR=../src/proto

# edit protoc to your own protobuf-compiler
protoc --proto_path=$PROTO_DIR $PROTO_DIR/HolidayCNN_proto.proto --cpp_out=$PROTO_DIR

