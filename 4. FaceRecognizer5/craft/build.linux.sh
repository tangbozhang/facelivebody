#!/usr/bin/env bash

cmake .. -DSEETA_LOCK_SDK_ONLINE=ON -DSEETA_LOCK_MODEL=ON -DSEETA_LOCK_KEY=$1

make -j16


