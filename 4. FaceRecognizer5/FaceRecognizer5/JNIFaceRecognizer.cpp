#include "JNIFaceRecognizer.h"
#include "jni_struct.hpp"
#include <VIPLFaceRecognizer.h>

#include <string>
#include <cstring>
#include <vector>
#include <iostream>
#include <memory>

using NativeClass = VIPLFaceRecognizer;

void JNICALL Java_com_seeta_sdk_FaceRecognizer_construct__Ljava_lang_String_2(JNIEnv* env, jobject self, jstring model)
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

void JNICALL Java_com_seeta_sdk_FaceRecognizer_construct__Ljava_lang_String_2Ljava_lang_String_2(JNIEnv* env, jobject self, jstring model, jstring device)
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

void JNICALL Java_com_seeta_sdk_FaceRecognizer_dispose(JNIEnv* env, jobject self)
{
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return;

	auto native_impl = reinterpret_cast<NativeClass *>(self_impl);
	delete native_impl;

	env->SetLongField(self, self_filed_impl, 0);
}


/*
* Class:     com_seeta_sdk_FaceRecognizer
* Method:    LoadModel
* Signature: (Ljava/lang/String;)Z
*/
jboolean JNICALL Java_com_seeta_sdk_FaceRecognizer_LoadModel__Ljava_lang_String_2(JNIEnv *env, jobject self, jstring model)
{
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return false;

	auto native_impl = reinterpret_cast<NativeClass *>(self_impl);

	auto native_model = jni_convert_string(env, model);

	return native_impl->LoadModel(native_model.c_str());
}

/*
* Class:     com_seeta_sdk_FaceRecognizer
* Method:    LoadModel
* Signature: (Ljava/lang/String;Ljava/lang/String;)Z
*/
jboolean JNICALL Java_com_seeta_sdk_FaceRecognizer_LoadModel__Ljava_lang_String_2Ljava_lang_String_2(JNIEnv* env, jobject self, jstring model, jstring device)
{
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return false;

	auto native_impl = reinterpret_cast<NativeClass *>(self_impl);

	auto native_model = jni_convert_string(env, model);
	auto native_device = jni_convert_string(env, device);

	NativeClass::Device type = NativeClass::AUTO;
	if (native_device == "CPU") type = NativeClass::CPU;
	else if (native_device == "GPU") type = NativeClass::GPU;

	return native_impl->LoadModel(native_model.c_str(), type);
}

/*
* Class:     com_seeta_sdk_FaceRecognizer
* Method:    GetFeatureSize
* Signature: ()I
*/
jint JNICALL Java_com_seeta_sdk_FaceRecognizer_GetFeatureSize(JNIEnv* env, jobject self)
{
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return 0;

	auto native_impl = reinterpret_cast<NativeClass *>(self_impl);

	return native_impl->GetFeatureSize();
}

/*
* Class:     com_seeta_sdk_FaceRecognizer
* Method:    GetCropWidth
* Signature: ()I
*/
jint JNICALL Java_com_seeta_sdk_FaceRecognizer_GetCropWidth(JNIEnv* env, jobject self)
{
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return 0;

	auto native_impl = reinterpret_cast<NativeClass *>(self_impl);

	return native_impl->GetCropWidth();
}

/*
* Class:     com_seeta_sdk_FaceRecognizer
* Method:    GetCropHeight
* Signature: ()I
*/
jint JNICALL Java_com_seeta_sdk_FaceRecognizer_GetCropHeight(JNIEnv* env, jobject self)
{
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return 0;

	auto native_impl = reinterpret_cast<NativeClass *>(self_impl);

	return native_impl->GetCropHeight();
}

/*
* Class:     com_seeta_sdk_FaceRecognizer
* Method:    GetCropChannels
* Signature: ()I
*/
jint JNICALL Java_com_seeta_sdk_FaceRecognizer_GetCropChannels(JNIEnv* env, jobject self)
{
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return 0;

	auto native_impl = reinterpret_cast<NativeClass *>(self_impl);

	return native_impl->GetCropChannels();
}

/*
* Class:     com_seeta_sdk_FaceRecognizer
* Method:    CropFace
* Signature: (Lcom/seeta/sdk/ImageData;[Lcom/seeta/sdk/Point;Lcom/seeta/sdk/ImageData;I)Z
*/
jboolean JNICALL Java_com_seeta_sdk_FaceRecognizer_CropFace
(JNIEnv *env, jobject self, jobject image, jobjectArray landmarks, jobject face, jint pos_num)
{
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return false;

	auto native_impl = reinterpret_cast<NativeClass *>(self_impl);

	VIPLPoint5 native_point5;
	VIPLImageData native_face(native_impl->GetCropWidth(), native_impl->GetCropHeight(), native_impl->GetCropChannels());
	std::unique_ptr<uint8_t[]> native_face_data(new uint8_t[native_face.width * native_face.height * native_face.channels]);
	native_face.data = native_face_data.get();
	auto native_image = jni_convert_image_data(env, image);
	jni_convert_point5(env, landmarks, native_point5);
	int native_pos_num = pos_num;

	auto result = native_impl->CropFace(native_image, native_point5, native_face, native_pos_num);

	if (result)
	{
		jni_set_image_data(env, native_face, face);
	}

	delete[] native_image.data;

	return result;
}

/*
* Class:     com_seeta_sdk_FaceRecognizer
* Method:    ExtractFeature
* Signature: (Lcom/seeta/sdk/ImageData;[F)Z
*/
jboolean JNICALL Java_com_seeta_sdk_FaceRecognizer_ExtractFeature__Lcom_seeta_sdk_ImageData_2_3F
(JNIEnv *env, jobject self, jobject face, jfloatArray feats)
{
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return false;

	auto native_impl = reinterpret_cast<NativeClass *>(self_impl);

	jsize feats_size = env->GetArrayLength(feats);
	size_t need_feats_size = native_impl->GetFeatureSize();
	if (feats_size < need_feats_size) jni_throw(env, "feats must have enough size");

	auto native_face = jni_convert_image_data(env, face);
	std::unique_ptr<float[]> native_feats(new float[need_feats_size]);

	auto result = native_impl->ExtractFeature(native_face, native_feats.get());

	if (result)
	{
		env->SetFloatArrayRegion(feats, 0, need_feats_size, native_feats.get());
	}

	delete[] native_face.data;

	return result;
}

/*
* Class:     com_seeta_sdk_FaceRecognizer
* Method:    ExtractFeatureNormalized
* Signature: (Lcom/seeta/sdk/ImageData;[F)Z
*/
jboolean JNICALL Java_com_seeta_sdk_FaceRecognizer_ExtractFeatureNormalized__Lcom_seeta_sdk_ImageData_2_3F
(JNIEnv *env, jobject self, jobject face, jfloatArray feats)
{
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return false;

	auto native_impl = reinterpret_cast<NativeClass *>(self_impl);

	jsize feats_size = env->GetArrayLength(feats);
	size_t need_feats_size = native_impl->GetFeatureSize();
	if (feats_size < need_feats_size) jni_throw(env, "feats must have enough size");

	auto native_face = jni_convert_image_data(env, face);
	std::unique_ptr<float[]> native_feats(new float[need_feats_size]);

	auto result = native_impl->ExtractFeatureNormalized(native_face, native_feats.get());

	if (result)
	{
		env->SetFloatArrayRegion(feats, 0, need_feats_size, native_feats.get());
	}

	delete[] native_face.data;

	return result;
}

/*
* Class:     com_seeta_sdk_FaceRecognizer
* Method:    ExtractFeatureWithCrop
* Signature: (Lcom/seeta/sdk/ImageData;[Lcom/seeta/sdk/Point;[FI)Z
*/
jboolean JNICALL Java_com_seeta_sdk_FaceRecognizer_ExtractFeatureWithCrop__Lcom_seeta_sdk_ImageData_2_3Lcom_seeta_sdk_Point_2_3FI
(JNIEnv *env, jobject self, jobject image, jobjectArray landmarks, jfloatArray feats, jint pos_num)
{
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return false;

	auto native_impl = reinterpret_cast<NativeClass *>(self_impl);

	jsize feats_size = env->GetArrayLength(feats);
	size_t need_feats_size = native_impl->GetFeatureSize();
	if (feats_size < need_feats_size) jni_throw(env, "feats must have enough size");

	VIPLPoint5 native_point5;
	auto native_image = jni_convert_image_data(env, image);
	jni_convert_point5(env, landmarks, native_point5);
	std::unique_ptr<float[]> native_feats(new float[need_feats_size]);
	int native_pos_num = pos_num;

	auto result = native_impl->ExtractFeatureWithCrop(native_image, native_point5, native_feats.get(), native_pos_num);

	if (result)
	{
		env->SetFloatArrayRegion(feats, 0, need_feats_size, native_feats.get());
	}

	delete[] native_image.data;

	return result;
}

/*
* Class:     com_seeta_sdk_FaceRecognizer
* Method:    ExtractFeatureWithCropNormalized
* Signature: (Lcom/seeta/sdk/ImageData;[Lcom/seeta/sdk/Point;[FI)Z
*/
jboolean JNICALL Java_com_seeta_sdk_FaceRecognizer_ExtractFeatureWithCropNormalized__Lcom_seeta_sdk_ImageData_2_3Lcom_seeta_sdk_Point_2_3FI
(JNIEnv *env, jobject self, jobject image, jobjectArray landmarks, jfloatArray feats, jint pos_num)
{
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return false;

	auto native_impl = reinterpret_cast<NativeClass *>(self_impl);

	jsize feats_size = env->GetArrayLength(feats);
	size_t need_feats_size = native_impl->GetFeatureSize();
	if (feats_size < need_feats_size) jni_throw(env, "feats must have enough size");

	VIPLPoint5 native_point5;
	auto native_image = jni_convert_image_data(env, image);
	jni_convert_point5(env, landmarks, native_point5);
	std::unique_ptr<float[]> native_feats(new float[need_feats_size]);
	int native_pos_num = pos_num;

	auto result = native_impl->ExtractFeatureWithCropNormalized(native_image, native_point5, native_feats.get(), native_pos_num);

	if (result)
	{
		env->SetFloatArrayRegion(feats, 0, need_feats_size, native_feats.get());
	}

	delete[] native_image.data;
	
	return result;
}

/*
* Class:     com_seeta_sdk_FaceRecognizer
* Method:    CalcSimilarity
* Signature: ([F[FJ)F
*/
jfloat JNICALL Java_com_seeta_sdk_FaceRecognizer_CalcSimilarity
(JNIEnv *env, jobject self, jfloatArray fc1, jfloatArray fc2, jlong dim)
{
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return false;

	auto native_impl = reinterpret_cast<NativeClass *>(self_impl);

	jfloat *fc1_ptr = env->GetFloatArrayElements(fc1, nullptr);
	jfloat *fc2_ptr = env->GetFloatArrayElements(fc2, nullptr);

	float *native_fc1 = fc1_ptr;
	float *native_fc2 = fc2_ptr;
	long native_dim = dim;

	float similar = native_impl->CalcSimilarity(native_fc1, native_fc2, native_dim);

	env->ReleaseFloatArrayElements(fc1, fc1_ptr, 0);
	env->ReleaseFloatArrayElements(fc2, fc2_ptr, 0);

	return similar;
}

/*
* Class:     com_seeta_sdk_FaceRecognizer
* Method:    CalcSimilarityNormalized
* Signature: ([F[FJ)F
*/
jfloat JNICALL Java_com_seeta_sdk_FaceRecognizer_CalcSimilarityNormalized
(JNIEnv *env, jobject self, jfloatArray fc1, jfloatArray fc2, jlong dim)
{
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return false;

	auto native_impl = reinterpret_cast<NativeClass *>(self_impl);

	jfloat *fc1_ptr = env->GetFloatArrayElements(fc1, nullptr);
	jfloat *fc2_ptr = env->GetFloatArrayElements(fc2, nullptr);

	float *native_fc1 = fc1_ptr;
	float *native_fc2 = fc2_ptr;
	long native_dim = dim;

	float similar =  native_impl->CalcSimilarityNormalized(native_fc1, native_fc2, native_dim);

	env->ReleaseFloatArrayElements(fc1, fc1_ptr, 0);
	env->ReleaseFloatArrayElements(fc2, fc2_ptr, 0);

	return similar;
}

JNIEXPORT void JNICALL Java_com_seeta_sdk_FaceRecognizer_SetNumThreads
  (JNIEnv *env, jobject self, jint num)
  {
	  // jclass self_class = env->GetObjectClass(self);
	// jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	// jlong self_impl = env->GetLongField(self, self_filed_impl);

	// if (!self_impl) return;

	// auto native_impl = reinterpret_cast<NativeClass *>(self_impl);
	// native_impl->SetNumThreads(num);
	NativeClass::SetNumThreads(num);
  }
  
  JNIEXPORT jint JNICALL Java_com_seeta_sdk_FaceRecognizer_SetMaxBatchGlobal
  (JNIEnv *env, jclass self, jint max_batch)
  {
	// jclass self_class = env->GetObjectClass(self);
	// jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	// jlong self_impl = env->GetLongField(self, self_filed_impl);

	// if (!self_impl) return -1;

	// auto native_impl = reinterpret_cast<NativeClass *>(self_impl);
	// auto result = native_impl->SetMaxBatchGlobal(max_batch);
	auto result = NativeClass::SetMaxBatchGlobal(max_batch);
	return result;
  }
  
  JNIEXPORT jint JNICALL Java_com_seeta_sdk_FaceRecognizer_GetMaxBatch
  (JNIEnv *env, jobject self)
  {
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return -1;

	auto native_impl = reinterpret_cast<NativeClass *>(self_impl);
	auto result = native_impl->GetMaxBatch();
	return result;
  }
  
  JNIEXPORT jint JNICALL Java_com_seeta_sdk_FaceRecognizer_SetCoreNumberGlobal
  (JNIEnv *env, jclass self, jint core_number)
  {
	// jclass self_class = env->GetObjectClass(self);
	// jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	// jlong self_impl = env->GetLongField(self, self_filed_impl);

	// if (!self_impl) return -1;

	// auto native_impl = reinterpret_cast<NativeClass *>(self_impl);
	// auto result = native_impl->SetCoreNumberGlobal(core_number);
	
	auto result = NativeClass::SetCoreNumberGlobal(core_number);
	return result;
  }
  
  JNIEXPORT jint JNICALL Java_com_seeta_sdk_FaceRecognizer_GetCoreNumber
  (JNIEnv *env, jobject self)
  {
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return -1;

	auto native_impl = reinterpret_cast<NativeClass *>(self_impl);
	auto result = native_impl->GetCoreNumber();
	return result;
  }
  
  JNIEXPORT jboolean JNICALL Java_com_seeta_sdk_FaceRecognizer_ExtractFeature___3Lcom_seeta_sdk_ImageData_2_3F
  (JNIEnv *env, jobject self, jobjectArray faces, jfloatArray feats)
  {
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return false;
	
    auto native_impl = reinterpret_cast<NativeClass *>(self_impl);
		
	auto faces_size = env->GetArrayLength(faces);
	auto native_faces = jni_convert_image_datas(env, faces);
	
	size_t need_feats_size = native_impl->GetFeatureSize() * faces_size;
	jsize feats_size = env->GetArrayLength(feats);
	if(feats_size < need_feats_size) jni_throw(env, "feats must have enough size");
	
	std::unique_ptr<float[]> native_feats(new float[need_feats_size]);
	
	auto result = native_impl->ExtractFeature(native_faces, native_feats.get());
	
	if (result)
	{
		env->SetFloatArrayRegion(feats, 0, need_feats_size, native_feats.get());
	}
	
	for(int i=0; i <faces_size; ++i)
	{
		delete[] native_faces[i].data;
	}
	
	return result;
  }
  
  JNIEXPORT jboolean JNICALL Java_com_seeta_sdk_FaceRecognizer_ExtractFeatureNormalized___3Lcom_seeta_sdk_ImageData_2_3F
  (JNIEnv *env, jobject self, jobjectArray faces, jfloatArray feats)
  {
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return false;
	
	auto native_impl = reinterpret_cast<NativeClass *>(self_impl);
		
	auto faces_size = env->GetArrayLength(faces);
	auto native_faces = jni_convert_image_datas(env, faces);
	
	size_t need_feats_size = native_impl->GetFeatureSize() * faces_size;
	jsize feats_size = env->GetArrayLength(feats);
	if(feats_size < need_feats_size) jni_throw(env, "feats must have enough size");
	
	std::unique_ptr<float[]> native_feats(new float[need_feats_size]);
	auto result = native_impl->ExtractFeatureNormalized(native_faces, native_feats.get());
	
	if (result)
	{
		env->SetFloatArrayRegion(feats, 0, need_feats_size, native_feats.get());
	}
	
	for(int i=0; i <faces_size; ++i)
	{
		delete[] native_faces[i].data;
	}
	
	return result;
  }
  
  JNIEXPORT jboolean JNICALL Java_com_seeta_sdk_FaceRecognizer_ExtractFeatureWithCrop___3Lcom_seeta_sdk_ImageData_2_3Lcom_seeta_sdk_Point_2_3F
  (JNIEnv *env, jobject self, jobjectArray images, jobjectArray landmarks, jfloatArray feats)
  {
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return false;
	
	auto native_impl = reinterpret_cast<NativeClass *>(self_impl);
	
	auto images_size = env->GetArrayLength(images);
	auto native_images = jni_convert_image_datas(env, images);
	
	size_t need_feats_size = native_impl->GetFeatureSize() * images_size;
	jsize feats_size = env->GetArrayLength(feats);
	if(feats_size < need_feats_size) jni_throw(env, "feats must have enough size");
	
	int N = 5;//5点
	auto landmarks_size = env->GetArrayLength(landmarks);
	size_t landmarks_need_size = images_size * N;
	if(landmarks_size < landmarks_need_size) jni_throw(env, "landmarks must have enough size");
	
	std::vector<VIPLPoint> native_points(landmarks_size);
	jni_convert_points(env, landmarks, native_points, landmarks_size);
	
    std::unique_ptr<float[]> native_feats(new float[need_feats_size]);
	auto result = native_impl->ExtractFeatureWithCrop(native_images, native_points, native_feats.get());
	
    if (result)
	{
		env->SetFloatArrayRegion(feats, 0, need_feats_size, native_feats.get());
	}
	
	for(int i=0; i <images_size; ++i)
	{
		delete[] native_images[i].data;
	}
	
	return result;
  }
  
  JNIEXPORT jboolean JNICALL Java_com_seeta_sdk_FaceRecognizer_ExtractFeatureWithCropNormalized___3Lcom_seeta_sdk_ImageData_2_3Lcom_seeta_sdk_Point_2_3F
  (JNIEnv *env, jobject self, jobjectArray images, jobjectArray landmarks, jfloatArray feats)
  {
	jclass self_class = env->GetObjectClass(self);
	jfieldID self_filed_impl = env->GetFieldID(self_class, "impl", "J");

	jlong self_impl = env->GetLongField(self, self_filed_impl);

	if (!self_impl) return false;
	
	auto native_impl = reinterpret_cast<NativeClass *>(self_impl);
	
	auto images_size = env->GetArrayLength(images);
	auto native_images = jni_convert_image_datas(env, images);
	
	size_t need_feats_size = native_impl->GetFeatureSize() * images_size;
	jsize feats_size = env->GetArrayLength(feats);
	if(feats_size < need_feats_size) jni_throw(env, "feats must have enough size");
	
	int N = 5;//5点
	auto landmarks_size = env->GetArrayLength(landmarks);
	size_t landmarks_need_size = images_size * N;
	if(landmarks_size < landmarks_need_size) jni_throw(env, "landmarks must have enough size");
	
	std::vector<VIPLPoint> native_points(landmarks_size);
	jni_convert_points(env, landmarks, native_points, landmarks_size);
	
    std::unique_ptr<float[]> native_feats(new float[need_feats_size]);
	auto result = native_impl->ExtractFeatureWithCropNormalized(native_images, native_points, native_feats.get());
	
    if (result)
	{
		env->SetFloatArrayRegion(feats, 0, need_feats_size, native_feats.get());
	}
	
	for(int i=0; i <images_size; ++i)
	{
		delete[] native_images[i].data;
	}
	
	return result;
  }