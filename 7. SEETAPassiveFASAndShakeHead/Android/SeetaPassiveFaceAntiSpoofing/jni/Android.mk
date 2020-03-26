LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := poseestimation-prebuilt
LOCAL_SRC_FILES := $(LOCAL_PATH)/../../../../5.PoseEstimation/PoseEstimation/libs/$(TARGET_ARCH_ABI)/libPoseEstimation.so
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/../../../../5.PoseEstimation/PoseEstimation/include/
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := eyeblinkdetect-prebuilt
LOCAL_SRC_FILES := $(LOCAL_PATH)/../../../../6.EyeBlinkDetector/EyeBlinkDetector/libs/$(TARGET_ARCH_ABI)/libEyeBlinkDetector.so
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/../../../../6.EyeBlinkDetector/EyeBlinkDetector/include/
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := seetanet-prebuilt
LOCAL_SRC_FILES := $(LOCAL_PATH)/../../../../1.SeetaNet/libs/$(TARGET_ARCH_ABI)/libseetanet.so
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/../../../../1.SeetaNet/include/
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := orz-prebuilt
LOCAL_SRC_FILES := $(LOCAL_PATH)/../../../../0.OpenRoleZoo/obj/local/$(TARGET_ARCH_ABI)/libOpenRoleZoo.a
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/../../../../0.OpenRoleZoo/include/
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)

LOCAL_MODULE := FaceSpoofingDetectJni

MY_CPP_LIST := $(wildcard $(LOCAL_PATH)/../*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../quality/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../seeta/*.cpp)

LOCAL_SRC_FILES := $(MY_CPP_LIST:$(LOCAL_PATH)/%=%)

LOCAL_C_INCLUDES += $(LOCAL_PATH)/../
LOCAL_C_INCLUDES += $(LOCAL_PATH)/../quality/
LOCAL_C_INCLUDES += $(LOCAL_PATH)/../seeta/

LOCAL_LDFLAGS += -L$(LOCAL_PATH)/lib

LOCAL_LDLIBS += -llog

ifeq ($(TARGET_ARCH_ABI), armeabi-v7a)
    LOCAL_CFLAGS += -mfloat-abi=softfp -mfpu=neon
endif

LOCAL_SHARED_LIBRARIES += eyeblinkdetect-prebuilt

LOCAL_SHARED_LIBRARIES += poseestimation-prebuilt

LOCAL_SHARED_LIBRARIES += seetanet-prebuilt

LOCAL_STATIC_LIBRARIES += orz-prebuilt

include $(BUILD_SHARED_LIBRARY)
