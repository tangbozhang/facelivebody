APP_STL := c++_static
APP_CPPFLAGS := -std=c++11
APP_CPPFLAGS += -fexceptions
APP_CPPFLAGS += -fPIC -frtti  -fpermissive #-Wl,--no-warn-mismatch -mhard-float -D_NDK_MATH_NO_SOFTFP=1 #
APP_ABI := armeabi-v7a arm64-v8a
APP_OPTIM := release
APP_PLATFORM := android-16
