#include "JNIPointDetector.h"
#include "jni_struct.hpp"
#include <VIPLPointDetector.h>

using NativeClass = VIPLPointDetector;

void JNICALL Java_com_seeta_sdk_PointDetector_construct(JNIEnv* env, jobject self, jstring model)
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

void JNICALL Java_com_seeta_sdk_PointDetector_dispose(JNIEnv* env, jobject self)
{
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return;

	auto native_impl = reinterpret_cast<NativeClass *>(self_impl);
	delete native_impl;

	env->SetLongField(self, self_filed_impl, 0);
}

void JNICALL Java_com_seeta_sdk_PointDetector_LoadModel(JNIEnv* env, jobject self, jstring model)
{
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return;

	auto native_impl = reinterpret_cast<NativeClass *>(self_impl);

	auto native_model = jni_convert_string(env, model);

	native_impl->LoadModel(native_model.c_str());
}

void JNICALL Java_com_seeta_sdk_PointDetector_SetStable(JNIEnv* env, jobject self, jboolean stable)
{
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return;

	auto native_impl = reinterpret_cast<NativeClass *>(self_impl);

	native_impl->SetStable(stable);
}

jint JNICALL Java_com_seeta_sdk_PointDetector_LandmarkNum(JNIEnv* env, jobject self)
{
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return 0;

	auto native_impl = reinterpret_cast<NativeClass *>(self_impl);

	return native_impl->LandmarkNum();
}

jboolean JNICALL Java_com_seeta_sdk_PointDetector_DetectCroppedLandmarks(JNIEnv* env, jobject self, jobject image, jobjectArray landmarks)
{
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return false;

	auto native_impl = reinterpret_cast<NativeClass *>(self_impl);

	jsize landmarks_size = env->GetArrayLength(landmarks);
	if (landmarks_size < native_impl->LandmarkNum()) jni_throw(env, "Landmarks must have enough length");

	auto native_image = jni_convert_image_data(env, image);

	std::vector<VIPLPoint> native_landmarks;
	native_impl->DetectCroppedLandmarks(native_image, native_landmarks);

	// convert native to java
	for (jsize i = 0; i < native_impl->LandmarkNum(); ++i)
	{
		jobject point = jni_convert(env, native_landmarks[i]);
		env->SetObjectArrayElement(landmarks, i, point);
	}

	delete[] native_image.data;

	return true;
}

jboolean JNICALL Java_com_seeta_sdk_PointDetector_DetectLandmarks(JNIEnv* env, jobject self, jobject image, jobject info, jobjectArray landmarks)
{
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return false;

	auto native_impl = reinterpret_cast<NativeClass *>(self_impl);

	jsize landmarks_size = env->GetArrayLength(landmarks);
	if (landmarks_size < native_impl->LandmarkNum()) jni_throw(env, "Landmarks must have enough length");

	auto native_image = jni_convert_image_data(env, image);
	auto native_info = jni_convert_face_info(env, info);

	std::vector<VIPLPoint> native_landmarks;
	native_impl->DetectLandmarks(native_image, native_info, native_landmarks);

	// convert native to java
	for (jsize i = 0; i < native_impl->LandmarkNum(); ++i)
	{
		jobject point = jni_convert(env, native_landmarks[i]);
		env->SetObjectArrayElement(landmarks, i, point);
	}

	delete[] native_image.data;

	return true;
}
