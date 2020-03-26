APP_STL := gnustl_static
APP_CPPFLAGS := -std=c++11

APP_CPPFLAGS += -fexceptions -O3

APP_CPPFLAGS += -fPIC -frtti -fvisibility=hidden

APP_ABI := armeabi-v7a arm64-v8a 
APP_PLATFORM := android-19
APP_OPTIM := release
