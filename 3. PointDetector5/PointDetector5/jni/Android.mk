LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := openrolezoo-prebuilt
LOCAL_SRC_FILES := $(LOCAL_PATH)/../../../0.OpenRoleZoo/obj/local/$(TARGET_ARCH_ABI)/libOpenRoleZoo.a
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/../../../0.OpenRoleZoo/include/
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)

LOCAL_MODULE := PointDetector500Jni

MY_CPP_LIST := $(wildcard $(LOCAL_PATH)/../*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../viplnet/src/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../viplnet/src/nets/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../viplnet/src/utils/log.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../viplnet/src/utils/math_functions_android.cpp)
LOCAL_SRC_FILES := $(MY_CPP_LIST:$(LOCAL_PATH)/%=%)

LOCAL_C_INCLUDES += $(LOCAL_PATH)/../
LOCAL_C_INCLUDES += $(LOCAL_PATH)/../include/
LOCAL_C_INCLUDES += $(LOCAL_PATH)/../viplnet/include/
LOCAL_C_INCLUDES += $(LOCAL_PATH)/../viplnet/include/nets/
LOCAL_C_INCLUDES += $(LOCAL_PATH)/../viplnet/include/util/

LOCAL_LDFLAGS += -L$(LOCAL_PATH)/lib #-fuse-ld=bfd

ifeq ($(TARGET_ARCH_ABI), armeabi-v7a)
    LOCAL_CFLAGS += -mfloat-abi=softfp -mfpu=neon
endif

LOCAL_LDLIBS += -llog -latomic -lm #-Wl,-lm_hard -Wl,--no-warn-mismatch #

LOCAL_STATIC_LIBRARIES += openrolezoo-prebuilt

include $(BUILD_SHARED_LIBRARY)
