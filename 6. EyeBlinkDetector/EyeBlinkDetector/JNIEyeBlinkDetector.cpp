#include "JNIEyeBlinkDetector.h"
#include "jni_struct.hpp"
#include <VIPLEyeBlinkDetector.h>

#include <string>
#include <cstring>
#include <vector>
#include <iostream>
#include <memory>

using NativeClass = VIPLEyeBlinkDetector;

JNIEXPORT void JNICALL Java_com_seeta_sdk_EyeBlinkDetector_construct__Ljava_lang_String_2
  (JNIEnv *env, jobject self, jstring model)
  {
	 jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	auto native_model = jni_convert_string(env, model);

	NativeClass *native_impl = nullptr;
	try
	{
		native_impl = new NativeClass(native_model.empty() ? nullptr : native_model.c_str());
	}
	catch (const std::exception &e)
	{
		jni_throw(env, e.what());
	}

	jlong self_impl = reinterpret_cast<jlong>(native_impl);

	env->SetLongField(self, self_filed_impl, self_impl);
  }
  
  JNIEXPORT void JNICALL Java_com_seeta_sdk_EyeBlinkDetector_construct__Ljava_lang_String_2Ljava_lang_String_2
  (JNIEnv *env, jobject self, jstring model, jstring device)
  {
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	auto native_model = jni_convert_string(env, model);
	auto native_device = jni_convert_string(env, device);

	NativeClass::Device type = NativeClass::AUTO;
	if (native_device == "CPU") type = NativeClass::CPU;
	else if (native_device == "GPU") type = NativeClass::GPU;

	NativeClass *native_impl = nullptr;
	try
	{
		native_impl = new NativeClass(native_model.c_str(), type);
	}
	catch(const std::exception &e)
	{
		jni_throw(env, e.what());
	}

	jlong self_impl = reinterpret_cast<jlong>(native_impl);

	env->SetLongField(self, self_filed_impl, self_impl);
  }
  
  JNIEXPORT void JNICALL Java_com_seeta_sdk_EyeBlinkDetector_dispose
  (JNIEnv *env, jobject self)
  {
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return;

	auto native_impl = reinterpret_cast<NativeClass *>(self_impl);
	delete native_impl;

	env->SetLongField(self, self_filed_impl, 0);
  }
  
  JNIEXPORT jint JNICALL Java_com_seeta_sdk_EyeBlinkDetector_Detect
  (JNIEnv *env, jobject self, jobject image, jobjectArray landmarks)
  {
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return -1;

	auto native_impl = reinterpret_cast<NativeClass *>(self_impl);

	typedef VIPLPoint VIPLPoint5[5];
	
	VIPLPoint5 native_point5;
	auto native_image = jni_convert_image_data(env, image);
	jni_convert_point5(env, landmarks, native_point5);
	
	int status = native_impl->Detect(native_image, native_point5);
	
	delete[] native_image.data;
	return status;
  }