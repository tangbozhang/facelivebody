#include "VIPLCFaceRecognizer.h"

#include "VIPLFaceRecognizer.h"

using VIPLCClass = VIPLFaceRecognizer;

VIPLCFaceRecognizer* VIPLCNewFaceRecognizer(const char* model_path, VIPLCDevice device)
{
	try
	{
		return reinterpret_cast<VIPLCFaceRecognizer *>(
			new VIPLCClass(model_path, VIPLCClass::Device(device)));
	}
	catch (...)
	{
		return nullptr;
	}
}

void VIPLCDeleteFaceRecognizer(const VIPLCFaceRecognizer* obj)
{
	if (obj == nullptr) return;
	auto impl = reinterpret_cast<const VIPLCClass *>(obj);
	delete impl;
}

int VIPLCFaceRecognizer_LoadModel(VIPLCFaceRecognizer* obj, const char* model_path, VIPLCDevice device)
{
	if (obj == nullptr) return 0;
	auto impl = reinterpret_cast<VIPLCClass *>(obj);

	return impl->LoadModel(model_path, VIPLCClass::Device(device));
}

uint32_t VIPLCFaceRecognizer_GetFeatureSize(VIPLCFaceRecognizer* obj)
{
	if (obj == nullptr) return 0;
	auto impl = reinterpret_cast<VIPLCClass *>(obj);

	return impl->GetFeatureSize();
}

uint32_t VIPLCFaceRecognizer_GetCropWidth(VIPLCFaceRecognizer* obj)
{
	if (obj == nullptr) return 0;
	auto impl = reinterpret_cast<VIPLCClass *>(obj);

	return impl->GetCropWidth();
}

uint32_t VIPLCFaceRecognizer_GetCropHeight(VIPLCFaceRecognizer* obj)
{
	if (obj == nullptr) return 0;
	auto impl = reinterpret_cast<VIPLCClass *>(obj);

	return impl->GetCropHeight();
}

uint32_t VIPLCFaceRecognizer_GetCropChannels(VIPLCFaceRecognizer* obj)
{
	if (obj == nullptr) return 0;
	auto impl = reinterpret_cast<VIPLCClass *>(obj);

	return impl->GetCropChannels();
}

int VIPLCFaceRecognizer_CropFace(VIPLCFaceRecognizer* obj, VIPLImageData image, const VIPLPoint* landmarks, VIPLImageData* cropped_face)
{
	if (obj == nullptr) return 0;
	auto impl = reinterpret_cast<VIPLCClass *>(obj);

	VIPLPoint points[5];
	for (int i = 0; i < 5; ++i) points[i] = landmarks[i];

	return impl->CropFace(image, points, *cropped_face);
}

int VIPLCFaceRecognizer_ExtractFeature(VIPLCFaceRecognizer* obj, VIPLImageData cropped_face, float* feats)
{
	if (obj == nullptr) return 0;
	auto impl = reinterpret_cast<VIPLCClass *>(obj);

	return impl->ExtractFeature(cropped_face, feats);
}

int VIPLCFaceRecognizer_ExtractFeatureWithCrop(VIPLCFaceRecognizer* obj, VIPLImageData image, const VIPLPoint* landmarks, float* feats)
{
	if (obj == nullptr) return 0;
	auto impl = reinterpret_cast<VIPLCClass *>(obj);

	VIPLPoint points[5];
	for (int i = 0; i < 5; ++i) points[i] = landmarks[i];

	return impl->ExtractFeatureWithCrop(image, points, feats);
}

float VIPLCFaceRecognizer_CalcSimilarity(VIPLCFaceRecognizer* obj, const float* fc1, const float* fc2)
{
	if (obj == nullptr) return 0;
	auto impl = reinterpret_cast<VIPLCClass *>(obj);

	return impl->CalcSimilarity(
		const_cast<float*>(fc1),
		const_cast<float*>(fc2));
}

int VIPLCFaceRecognizer_ExtractFeatureNormalized(VIPLCFaceRecognizer* obj, VIPLImageData cropped_face, float* feats)
{
	if (obj == nullptr) return 0;
	auto impl = reinterpret_cast<VIPLCClass *>(obj);

	return impl->ExtractFeatureNormalized(cropped_face, feats);
}

int VIPLCFaceRecognizer_ExtractFeatureWithCropNormalized(VIPLCFaceRecognizer* obj, VIPLImageData image, const VIPLPoint* landmarks, float* feats)
{
	if (obj == nullptr) return 0;
	auto impl = reinterpret_cast<VIPLCClass *>(obj);

	VIPLPoint points[5];
	for (int i = 0; i < 5; ++i) points[i] = landmarks[i];

	return impl->ExtractFeatureWithCropNormalized(image, points, feats);
}

float VIPLCFaceRecognizer_CalcSimilarityNormalized(VIPLCFaceRecognizer* obj, const float* fc1, const float* fc2)
{
	if (obj == nullptr) return 0;
	auto impl = reinterpret_cast<VIPLCClass *>(obj);

	return impl->CalcSimilarityNormalized(
		const_cast<float*>(fc1),
		const_cast<float*>(fc2));
}

void VIPLCFaceRecognizer_SetNumThreads(int num)
{
	VIPLFaceRecognizer::SetNumThreads(num);
}
