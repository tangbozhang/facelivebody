rm CMakeCache.txt
make clean -j4
cmake . -DCMAKE_TOOLCHAIN_FILE=./iOS.cmake \
-DHOLIDAY_INCLUDE_DIRS=../SeetaNet/sources/release/include/ \
-DEBD_INCLUDE_DIRS=../EyeBlinkDetector/release/include/ \
-DPE_INCLUDE_DIRS=../PoseEstimation/release/include/ \
-DFR_INCLUDE_DIRS=../FaceRecognizer/release/include/ \
-DIOS_PLATFORM=SIMULATOR64
make -j4
