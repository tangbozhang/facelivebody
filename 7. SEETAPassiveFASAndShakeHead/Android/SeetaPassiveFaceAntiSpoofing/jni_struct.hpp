#include <jni.h>

#include "VIPLStruct.h"

#include <string>
#include <vector>
#include <memory>
#include <cstring>

static void jni_throw(JNIEnv* env, const std::string &msg)
{
	jclass Exception_class = env->FindClass("java/lang/Exception");
	env->ThrowNew(Exception_class, msg.c_str());
}

std::string jni_convert_string(JNIEnv *env, jstring jstr)
{
	// const char *native_str = env->GetStringUTFChars(str, nullptr);
	// std::string result = native_str;
	// env->ReleaseStringUTFChars(str, native_str);
	// return native_str;
	std::unique_ptr<char[]> cstr;
	jclass clsstring = env->FindClass("java/lang/String");
	jstring strencode = env->NewStringUTF("utf-8");
	jmethodID mid = env->GetMethodID(clsstring, "getBytes", "(Ljava/lang/String;)[B");
	jbyteArray barr = static_cast<jbyteArray>(env->CallObjectMethod(jstr, mid, strencode));
	jsize alen = env->GetArrayLength(barr);
	jbyte* ba = env->GetByteArrayElements(barr, JNI_FALSE);
	if (alen > 0)
	{
		cstr.reset(new char[alen + 1]);
		memcpy(cstr.get(), ba, alen);
		cstr[alen] = '\0';
	}
	env->ReleaseByteArrayElements(barr, ba, 0);
	env->DeleteLocalRef(strencode);
	env->DeleteLocalRef(barr);
	return cstr.get();
}
template <typename T>
static void CopyData(T *dst, const T *src, size_t count)
{
#if _MSC_VER >= 1600
	memcpy_s(dst, count * sizeof(T), src, count * sizeof(T));
#else
	memcpy(dst, src, count * sizeof(T));
#endif
}

static VIPLImageData jni_convert_image_data(JNIEnv* env, jobject image)
{
	jclass ImageData_class = env->GetObjectClass(image);
	jfieldID ImageData_field_data = env->GetFieldID(ImageData_class, "data", "[B");
	jfieldID ImageData_field_width = env->GetFieldID(ImageData_class, "width", "I");
	jfieldID ImageData_field_height = env->GetFieldID(ImageData_class, "height", "I");
	jfieldID ImageData_field_channels = env->GetFieldID(ImageData_class, "channels", "I");

	jbyteArray image_data = static_cast<jbyteArray>(env->GetObjectField(image, ImageData_field_data));
	jint image_width = env->GetIntField(image, ImageData_field_width);
	jint image_height = env->GetIntField(image, ImageData_field_height);
	jint image_channels = env->GetIntField(image, ImageData_field_channels);

	jsize image_data_size = env->GetArrayLength(image_data);
	size_t need_data_size = image_width * image_height * image_channels;

	if (image_data_size < need_data_size) jni_throw(env, "Miss match image data size.");

	VIPLImageData native_image(image_width, image_height, image_channels);

	native_image.data = new uint8_t[need_data_size];
	env->GetByteArrayRegion(image_data, 0, need_data_size, reinterpret_cast<jbyte*>(native_image.data));

	return native_image;
}

static jobject jni_convert_image_data(JNIEnv *env, const VIPLImageData &native_image)
{
	jclass ImageData_class = env->FindClass("com/seeta/sdk/ImageData");
	jmethodID ImageData_ctor = env->GetMethodID(ImageData_class, "<init>", "(III)V");
	jfieldID ImageData_field_data = env->GetFieldID(ImageData_class, "data", "[B");

	jobject jimage = env->NewObject(ImageData_class, ImageData_ctor, native_image.width, native_image.height, native_image.channels);

	size_t native_image_data_size = native_image.width * native_image.height * native_image.channels;

	jbyteArray jimage_data = static_cast<jbyteArray>(env->GetObjectField(jimage, ImageData_field_data));

	env->SetByteArrayRegion(jimage_data, 0, native_image_data_size, reinterpret_cast<const jbyte*>(native_image.data));

	return jimage;
}

static jobject jni_convert(JNIEnv *env, const VIPLFaceInfo &native_info)
{
	jclass FaceInfo_class = env->FindClass("com/seeta/sdk/FaceInfo");
	jfieldID FaceInfo_field_x = env->GetFieldID(FaceInfo_class, "x", "I");
	jfieldID FaceInfo_field_y = env->GetFieldID(FaceInfo_class, "y", "I");
	jfieldID FaceInfo_field_width = env->GetFieldID(FaceInfo_class, "width", "I");
	jfieldID FaceInfo_field_height = env->GetFieldID(FaceInfo_class, "height", "I");
	jfieldID FaceInfo_field_roll = env->GetFieldID(FaceInfo_class, "roll", "D");
	jfieldID FaceInfo_field_pitch = env->GetFieldID(FaceInfo_class, "pitch", "D");
	jfieldID FaceInfo_field_yaw = env->GetFieldID(FaceInfo_class, "yaw", "D");
	jfieldID FaceInfo_field_score = env->GetFieldID(FaceInfo_class, "score", "D");
	jmethodID FaceInfo_method_ctor = env->GetMethodID(FaceInfo_class, "<init>", "()V");

	jobject info = env->NewObject(FaceInfo_class, FaceInfo_method_ctor);
	env->SetIntField(info, FaceInfo_field_x, native_info.x);
	env->SetIntField(info, FaceInfo_field_y, native_info.y);
	env->SetIntField(info, FaceInfo_field_width, native_info.width);
	env->SetIntField(info, FaceInfo_field_height, native_info.height);
	env->SetDoubleField(info, FaceInfo_field_roll, native_info.roll);
	env->SetDoubleField(info, FaceInfo_field_pitch, native_info.pitch);
	env->SetDoubleField(info, FaceInfo_field_yaw, native_info.yaw);
	env->SetDoubleField(info, FaceInfo_field_score, native_info.score);

	return info;
}

static jobjectArray jni_convert(JNIEnv *env, const std::vector<VIPLFaceInfo> &native_infos)
{
	jclass FaceInfo_class = env->FindClass("com/seeta/sdk/FaceInfo");

	jsize infos_size = native_infos.size();
	jobjectArray infos = env->NewObjectArray(infos_size, FaceInfo_class, nullptr);

	for (jsize i = 0; i < infos_size; ++i)
	{
		jobject info = jni_convert(env, native_infos[i]);
		env->SetObjectArrayElement(infos, i, info);
	}

	return infos;
}

static VIPLFaceInfo jni_convert_face_info(JNIEnv* env, jobject info)
{
	jclass FaceInfo_class = env->GetObjectClass(info);
	jfieldID FaceInfo_field_x = env->GetFieldID(FaceInfo_class, "x", "I");
	jfieldID FaceInfo_field_y = env->GetFieldID(FaceInfo_class, "y", "I");
	jfieldID FaceInfo_field_width = env->GetFieldID(FaceInfo_class, "width", "I");
	jfieldID FaceInfo_field_height = env->GetFieldID(FaceInfo_class, "height", "I");
	jfieldID FaceInfo_field_roll = env->GetFieldID(FaceInfo_class, "roll", "D");
	jfieldID FaceInfo_field_pitch = env->GetFieldID(FaceInfo_class, "pitch", "D");
	jfieldID FaceInfo_field_yaw = env->GetFieldID(FaceInfo_class, "yaw", "D");
	jfieldID FaceInfo_field_score = env->GetFieldID(FaceInfo_class, "score", "D");

	VIPLFaceInfo native_info;

	native_info.x = env->GetIntField(info, FaceInfo_field_x);
	native_info.y = env->GetIntField(info, FaceInfo_field_y);
	native_info.width = env->GetIntField(info, FaceInfo_field_width);
	native_info.height = env->GetIntField(info, FaceInfo_field_height);
	native_info.roll = env->GetDoubleField(info, FaceInfo_field_roll);
	native_info.pitch = env->GetDoubleField(info, FaceInfo_field_pitch);
	native_info.yaw = env->GetDoubleField(info, FaceInfo_field_yaw);
	native_info.score = env->GetDoubleField(info, FaceInfo_field_score);

	return native_info;
}

static jobject jni_convert(JNIEnv *env, const VIPLPoint &native_point)
{
	jclass Point_class = env->FindClass("com/seeta/sdk/Point");
	jfieldID Point_field_x = env->GetFieldID(Point_class, "x", "D");
	jfieldID Point_field_y = env->GetFieldID(Point_class, "y", "D");
	jmethodID Point_method_ctor = env->GetMethodID(Point_class, "<init>", "()V");

	jobject point = env->NewObject(Point_class, Point_method_ctor);
	env->SetDoubleField(point, Point_field_x, native_point.x);
	env->SetDoubleField(point, Point_field_y, native_point.y);

	return point;
}

static void jni_set_image_data(JNIEnv* env, const VIPLImageData &native_image, jobject image)
{
	jclass ImageData_class = env->GetObjectClass(image);
	jfieldID ImageData_field_data = env->GetFieldID(ImageData_class, "data", "[B");
	jfieldID ImageData_field_width = env->GetFieldID(ImageData_class, "width", "I");
	jfieldID ImageData_field_height = env->GetFieldID(ImageData_class, "height", "I");
	jfieldID ImageData_field_channels = env->GetFieldID(ImageData_class, "channels", "I");

	jbyteArray image_data = static_cast<jbyteArray>(env->GetObjectField(image, ImageData_field_data));
	jint image_width = env->GetIntField(image, ImageData_field_width);
	jint image_height = env->GetIntField(image, ImageData_field_height);
	jint image_channels = env->GetIntField(image, ImageData_field_channels);

	jsize image_data_size = env->GetArrayLength(image_data);
	size_t native_image_data_size = native_image.width * native_image.height * native_image.channels;

	if (image_data_size < native_image_data_size) jni_throw(env, "Miss match image data size.");

	env->SetIntField(image, ImageData_field_width, native_image.width);
	env->SetIntField(image, ImageData_field_height, native_image.height);
	env->SetIntField(image, ImageData_field_channels, native_image.channels);

	env->SetByteArrayRegion(image_data, 0, native_image_data_size, reinterpret_cast<const jbyte*>(native_image.data));
}

static VIPLPoint jni_convert_point(JNIEnv *env, jobject point)
{
	jclass Point_class = env->GetObjectClass(point);
	jfieldID Point_field_x = env->GetFieldID(Point_class, "x", "D");
	jfieldID Point_field_y = env->GetFieldID(Point_class, "y", "D");

	VIPLPoint native_point;

	native_point.x = env->GetDoubleField(point, Point_field_x);
	native_point.y = env->GetDoubleField(point, Point_field_y);

	return native_point;
}

static std::vector<VIPLPoint> jni_convert_landmarks(JNIEnv *env, jobjectArray landmarks)
{
	jsize landmarks_size = env->GetArrayLength(landmarks);
	std::vector<VIPLPoint> native_landmarks(landmarks_size);
	for (jsize i = 0; i < landmarks_size; ++i)
	{
		jobject point = env->GetObjectArrayElement(landmarks, i);
		native_landmarks[i] = jni_convert_point(env, point);
	}
	return native_landmarks;
}

static void jni_convert_point5(JNIEnv *env, jobjectArray landmarks, VIPLPoint native_point5[])
{
	jsize landmarks_size = env->GetArrayLength(landmarks);
	if (landmarks_size < 5) jni_throw(env, "Landmarks must have 5 points");
	for (jsize i = 0; i < 5; ++i)
	{
		jobject point = env->GetObjectArrayElement(landmarks, i);
		native_point5[i] = jni_convert_point(env, point);
	}
}
