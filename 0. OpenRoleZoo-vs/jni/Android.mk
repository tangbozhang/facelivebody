LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := OpenRoleZoo

MY_CPP_LIST := $(wildcard $(LOCAL_PATH)/../*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/orz/codec/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/orz/io/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/orz/io/jug/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/orz/io/stream/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/orz/lego/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/orz/mem/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/orz/net/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/orz/ssl/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/orz/sync/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/orz/tools/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/orz/utils/*.cpp)

LOCAL_SRC_FILES := $(MY_CPP_LIST:$(LOCAL_PATH)/%=%)

LOCAL_C_INCLUDES += $(LOCAL_PATH)/../include/

LOCAL_LDFLAGS += -L$(LOCAL_PATH)/lib -fuse-ld=bfd

LOCAL_LDLIBS += -llog

include $(BUILD_STATIC_LIBRARY)
