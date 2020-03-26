LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := openblas
LOCAL_SRC_FILES := $(LOCAL_PATH)/../thirdParty/Android/openblas/libs/$(TARGET_ARCH_ABI)/libopenblas.a
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/../thirdParty/Android/openblas/include
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := protobuf
LOCAL_SRC_FILES := $(LOCAL_PATH)/../thirdParty/Android/protobuf/libs/$(TARGET_ARCH_ABI)/libprotobuf.a
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/../thirdParty/Android/protobuf/include
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)

LOCAL_MODULE := seetanet

LOCAL_CFLAGS += -DANDROID_PLATFORM
LOCAL_CFLAGS += -DSEETA_EXPORTS

MY_CPP_LIST := $(wildcard $(LOCAL_PATH)/../src/*.cpp)

MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/orz/mem/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/orz/sync/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/orz/tools/*.cpp)

MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/proto/*.cc)

LOCAL_SRC_FILES := $(MY_CPP_LIST:$(LOCAL_PATH)/%=%)

LOCAL_C_INCLUDES += $(LOCAL_PATH)/../include \
                    $(LOCAL_PATH)/../src/include_inner \
					$(LOCAL_PATH)/../src \
					$(LOCAL_PATH)/../src/proto \
                    $(LOCAL_PATH)/../thirdParty/Android/include/ \
                    $(LOCAL_PATH)/../thirdParty/Android/protobuf/include \
                    $(LOCAL_PATH)/../thirdParty/Android/openblas/include \
					$(LOCAL_PATH)/../thirdParty/Android/openblas/include/openblas

LOCAL_LDFLAGS += -L$(LOCAL_PATH)/lib -fuse-ld=bfd.exe

ifeq ($(TARGET_ARCH_ABI), armeabi-v7a)
    LOCAL_CFLAGS += -mfloat-abi=softfp -mfpu=neon
endif

LOCAL_LDLIBS += -lc -llog -latomic -lm

LOCAL_STATIC_LIBRARIES += protobuf openblas

include $(BUILD_SHARED_LIBRARY)
