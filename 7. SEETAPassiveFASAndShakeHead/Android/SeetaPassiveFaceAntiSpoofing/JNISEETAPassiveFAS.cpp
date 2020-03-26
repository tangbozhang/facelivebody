#include "JNISEETAPassiveFAS.h"
#include "jni_struct.hpp"
#include <SEETAPassiveFASAndShakeHead.h>

#include <string>
#include <cstring>
#include <vector>
#include <iostream>
#include <memory>

#include "android/log.h"
#define LOG_TAG "FASJni"
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG,  __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using NativeClass = SEETAPassiveFASAndShakeHead;

void JNICALL Java_com_seeta_sdk_SEETAPassiveFAS_construct
  (JNIEnv *env, jobject self, jstring model, jint shakeHeadAngleThresh, jint nodheadAngleThresh,
  jdouble clarityThresh, jdouble fuseThresh, jint systemStateIndex, jint firstPhaseFrameNum, jint detectFrameNum)
  {
	  jclass self_class = env->GetObjectClass(self);
	  jfieldID self_field_impl = env->GetFieldID(self_class, "impl", "J");
	  
	  auto native_model = jni_convert_string(env, model);
	  
	  NativeClass * native_impl = nullptr;
	  
	  int native_index = systemStateIndex;
	  SystemState state = SystemState(native_index);
	  try
	  {
		  native_impl = new NativeClass(native_model, shakeHeadAngleThresh, nodheadAngleThresh,
								clarityThresh, fuseThresh, state, firstPhaseFrameNum, detectFrameNum);
	  }
	  catch(const std::exception &e)
	  {
		  jni_throw(env, e.what());
		  return;
	  }
	  
	  jlong self_impl = reinterpret_cast<jlong>(native_impl);
	  
	  env->SetLongField(self, self_field_impl, self_impl);
  }
  
 int last_state_index = -1;//上一次活体的状态
  
jint JNICALL Java_com_seeta_sdk_SEETAPassiveFAS_DetectCore
 (JNIEnv *env, jobject self, jobject image, jobject info, jobjectArray points)
 {
		  
	  jclass self_class = env->GetObjectClass(self);
	  jfieldID self_field_impl = env->GetFieldID(self_class, "impl", "J");
	  
	  jlong self_impl = env->GetLongField(self, self_field_impl);
	  
	  if(!self_impl) return -1;
	  
	  auto native_impl = reinterpret_cast<NativeClass *>(self_impl);
	  
	  auto native_image = jni_convert_image_data(env, image);

	  auto native_info = jni_convert_face_info(env, info);

	  typedef VIPLPoint VIPLPoint5[5];
	  VIPLPoint5 native_point5;
	  jni_convert_point5(env, points, native_point5);
	  
	  int N = 5;
	  
	  
	  VIPLPoint *points5 = new VIPLPoint[N]; 
      for(int i=0;i<N;++i)
	  {
		  points5[i].x = native_point5[i].x;
		  points5[i].y = native_point5[i].y;
	  }		  
	  
	  SystemState result;
	  if(last_state_index == -1)
	  {
		  result = noFace;
	  }
	  else
	  {
		  result = SystemState(last_state_index);
	  }
	  native_impl->detecion(native_image, native_info, points5, result);
	  
	  delete[] native_image.data;
	  
	  delete[] points5;
	  points5 = nullptr;
	  
	  
	  last_state_index = (int)result;
	  jint state_index = (int)result;
	  
	  return state_index;
 }
  
void JNICALL Java_com_seeta_sdk_SEETAPassiveFAS_ResetCore (JNIEnv *env, jobject self, jint stateIndex)
 {
	 jclass self_class = env->GetObjectClass(self);
	 jfieldID self_field_impl = env->GetFieldID(self_class, "impl", "J");
	  
	 jlong self_impl = env->GetLongField(self, self_field_impl);
	  
	 if(!self_impl) return;
	  
	 auto native_impl = reinterpret_cast<NativeClass *>(self_impl);
	 
	 SystemState state = SystemState(stateIndex);
	 
	 native_impl->reset(state);
	 
	 last_state_index = -1;
 }
 
 jboolean JNICALL Java_com_seeta_sdk_SEETAPassiveFAS_SetActionsCore
 (JNIEnv *env, jobject self, jintArray stateIndexArray)
 {
	 jclass self_class = env->GetObjectClass(self);
	 jfieldID self_field_impl = env->GetFieldID(self_class, "impl", "J");
	  
	 jlong self_impl = env->GetLongField(self, self_field_impl);
	  
	 if(!self_impl) return false;
	  
	 auto native_impl = reinterpret_cast<NativeClass *>(self_impl);
	 
	 jint state_indexarray_size = env->GetArrayLength(stateIndexArray);
	 int* native_index_array = env->GetIntArrayElements(stateIndexArray, 0);
	 
	 std::vector<SystemState> actions;
	 for(int i = 0; i < state_indexarray_size; ++i)
	 {
		 int native_index = native_index_array[i];
		 actions.push_back(SystemState(native_index));
	 }
	 
	 jboolean result = native_impl->set_actions(actions);
	 env->ReleaseIntArrayElements(stateIndexArray, native_index_array, 0);
	 
	 return result;

 }
 
 void JNICALL Java_com_seeta_sdk_SEETAPassiveFAS_getLog
  (JNIEnv *env, jobject self, jdoubleArray value_array)
  {
	  jclass self_class = env->GetObjectClass(self);
	  jfieldID self_field_impl = env->GetFieldID(self_class, "impl", "J");
	  
	  jlong self_impl = env->GetLongField(self, self_field_impl);
	  
	  if(!self_impl) return;
	  
	  auto native_impl = reinterpret_cast<NativeClass *>(self_impl);
	  
	  jint value_array_size = env->GetArrayLength(value_array);
	  if(value_array_size != 4) return;
	  
	  jdouble jdouble_array[4];
	  env->GetDoubleArrayRegion(value_array, 0, 4, jdouble_array);
	  native_impl->getLog(jdouble_array[0], jdouble_array[1], jdouble_array[2], jdouble_array[3]);
	  
	  env->SetDoubleArrayRegion(value_array, 0, 4, jdouble_array);
	  
  }
  
void JNICALL Java_com_seeta_sdk_SEETAPassiveFAS_dispose(JNIEnv *env, jobject self)
 {
	  jclass self_class = env->GetObjectClass(self);
	  jfieldID self_field_impl = env->GetFieldID(self_class, "impl", "J");
	  
	  jlong self_impl = env->GetLongField(self, self_field_impl);
	  
	  if(!self_impl) return;
	  
	  auto native_impl = reinterpret_cast<NativeClass *>(self_impl);
	  delete native_impl;
	  
	  env->SetLongField(self, self_field_impl, 0);
 }
 
   JNIEXPORT jobject JNICALL Java_com_seeta_sdk_SEETAPassiveFAS_getSnapshot
  (JNIEnv *env, jobject self)
  {
	  jclass self_class = env->GetObjectClass(self);
	  jfieldID self_field_impl = env->GetFieldID(self_class, "impl", "J");
	  
	  jlong self_impl = env->GetLongField(self, self_field_impl);
	  
	  if(!self_impl) return nullptr;
	  
	  auto native_impl = reinterpret_cast<NativeClass *>(self_impl);
	  
	  VIPLImageData native_image = native_impl->getSnapshot();
	  jobject snapshot = jni_convert_image_data(env, native_image);
	  
	  return snapshot;
  }