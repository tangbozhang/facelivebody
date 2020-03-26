#include "JNIFaceDetector.h"
#include "jni_struct.hpp"
#include <VIPLFaceDetector.h>

#include <iostream>

using NativeClass = VIPLFaceDetector;

void JNICALL Java_com_seeta_sdk_FaceDetector_construct__Ljava_lang_String_2(JNIEnv* env, jobject self,
																		jstring model)
{
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	auto native_model = jni_convert_string(env, model);

	NativeClass *native_impl = nullptr;
	try
	{
		native_impl = new NativeClass(native_model.c_str());
	}
	catch (const std::exception &e)
	{
		jni_throw(env, e.what());
	}
	
	jlong self_impl = reinterpret_cast<jlong>(native_impl);

	env->SetLongField(self, self_filed_impl, self_impl);
}

void JNICALL Java_com_seeta_sdk_FaceDetector_construct__Ljava_lang_String_2Ljava_lang_String_2(JNIEnv* env, jobject self,
																					jstring model, jstring device)
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
	catch (const std::exception &e)
	{
		jni_throw(env, e.what());
	}

	jlong self_impl = reinterpret_cast<jlong>(native_impl);

	env->SetLongField(self, self_filed_impl, self_impl);
}

void JNICALL Java_com_seeta_sdk_FaceDetector_dispose(JNIEnv* env,jobject self)
{
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return;

	auto native_impl = reinterpret_cast<VIPLFaceDetector *>(self_impl);
	delete native_impl;

	env->SetLongField(self, self_filed_impl, 0);
}

jobjectArray JNICALL Java_com_seeta_sdk_FaceDetector_Detect(JNIEnv* env, jobject self,
										jobject image)
{
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return jni_convert(env, std::vector<VIPLFaceInfo>());

	auto native_impl = reinterpret_cast<VIPLFaceDetector *>(self_impl);

	auto native_image = jni_convert_image_data(env, image);

	auto native_infos = native_impl->Detect(native_image);

	auto infos = jni_convert(env, native_infos);

	delete[] native_image.data;

	return infos;
}

void JNICALL Java_com_seeta_sdk_FaceDetector_SetMinFaceSize(JNIEnv* env, jobject self,
													 jint size)
{
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return;

	auto native_impl = reinterpret_cast<VIPLFaceDetector *>(self_impl);

	native_impl->SetMinFaceSize(size);
}

void JNICALL Java_com_seeta_sdk_FaceDetector_SetImagePyramidScaleFactor(JNIEnv* env, jobject self,
																 jfloat factor)
{
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return;

	auto native_impl = reinterpret_cast<VIPLFaceDetector *>(self_impl);

	native_impl->SetImagePyramidScaleFactor(factor);
}

void JNICALL Java_com_seeta_sdk_FaceDetector_SetScoreThresh(JNIEnv* env, jobject self,
										 jfloat thresh1, jfloat thresh2, jfloat thresh3)
{
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return;

	auto native_impl = reinterpret_cast<VIPLFaceDetector *>(self_impl);

	native_impl->SetScoreThresh(thresh1, thresh2, thresh3);
}

void JNICALL Java_com_seeta_sdk_FaceDetector_SetVideoStable(JNIEnv* env, jobject self,
												 jboolean stable)
{
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return;

	auto native_impl = reinterpret_cast<VIPLFaceDetector *>(self_impl);

	native_impl->SetVideoStable(stable);
}

jboolean JNICALL Java_com_seeta_sdk_FaceDetector_GetVideoStable(JNIEnv* env, jobject self)
{
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return false;

	auto native_impl = reinterpret_cast<VIPLFaceDetector *>(self_impl);

	return native_impl->GetVideoStable();
}

JNIEXPORT void JNICALL Java_com_seeta_sdk_FaceDetector_SetNumThreads
  (JNIEnv *env, jobject self, jint num)
  {
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return;

	auto native_impl = reinterpret_cast<VIPLFaceDetector *>(self_impl);
	
	native_impl->SetNumThreads(num);
  }
