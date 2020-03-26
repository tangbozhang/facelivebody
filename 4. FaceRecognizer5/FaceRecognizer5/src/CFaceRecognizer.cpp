#include "seeta/CFaceRecognizer.h"
#include "seeta/FaceRecognizer.h"

struct SeetaFaceRecognizer
{
	seeta::FaceRecognizer *impl = nullptr;
};

SeetaFaceRecognizer* SeetaNewFaceRecognizer(SeetaModelSetting setting)
{
	std::unique_ptr<SeetaFaceRecognizer> obj(new SeetaFaceRecognizer);
	obj->impl = new seeta::FaceRecognizer(setting);
	return obj.release();
}

void SeetaDeleteFaceRecognizer(const SeetaFaceRecognizer* obj)
{
	if (!obj) return;
	delete obj->impl;
	delete obj;
}

int SeetaFaceRecognizer_SetLogLevel(int level)
{
    return seeta::FaceRecognizer::SetLogLevel(level);
}

const SeetaFaceRecognizer_SharedModel* SeetaFaceRecognizer_LoadModel(SeetaModelSetting setting)
{
    auto shared_model = reinterpret_cast<const SeetaFaceRecognizer_SharedModel*>(
        seeta::FaceRecognizer::LoadModel(setting));
    return shared_model;
}

void SeetaFaceRecognizer_FreeModel(const SeetaFaceRecognizer_SharedModel* model)
{
    auto shared_model = reinterpret_cast<const seeta::FaceRecognizer::SharedModel*>(model);
    seeta::FaceRecognizer::FreeModel(shared_model);
}

int SeetaFaceRecognizer_GetCropFaceWidth(const SeetaFaceRecognizer* obj)
{
    if (!obj || !obj->impl) return 0;
    return obj->impl->GetCropFaceWidth();
}

int SeetaFaceRecognizer_GetCropFaceHeight(const SeetaFaceRecognizer* obj)
{
    if (!obj || !obj->impl) return 0;
    return obj->impl->GetCropFaceHeight();
}

int SeetaFaceRecognizer_GetCropFaceChannels(const SeetaFaceRecognizer* obj)
{
    if (!obj || !obj->impl) return 0;
    return obj->impl->GetCropFaceChannels();
}

int SeetaFaceRecognizer_GetExtractFeatureSize(const SeetaFaceRecognizer* obj)
{
    if (!obj || !obj->impl) return 0;
    return obj->impl->GetExtractFeatureSize();
}

int SeetaFaceRecognizer_CropFace(const SeetaFaceRecognizer* obj, const SeetaImageData image, const SeetaPointF* points,
    SeetaImageData* face)
{
    if (!obj || !obj->impl) return 0;
    if (!points || !face) return 0;
    return int(obj->impl->CropFace(image, points, *face));
}

int SeetaFaceRecognizer_ExtractCroppedFace(const SeetaFaceRecognizer* obj, const SeetaImageData image, float* features)
{
    if (!obj || !obj->impl) return 0;
    if (!features) return 0;
    return int(obj->impl->ExtractCroppedFace(image, features));
}

int SeetaFaceRecognizer_Extract(const SeetaFaceRecognizer* obj, const SeetaImageData image, const SeetaPointF* points,
    float* features)
{
    if (!obj || !obj->impl) return 0;
    if (!points || !features) return 0;
    return int(obj->impl->Extract(image, points, features));
}

float SeetaFaceRecognizer_CalculateSimilarity(const SeetaFaceRecognizer* obj, const float* features1,
    const float* features2)
{
    if (!obj || !obj->impl) return 0;
    if (!features1 || !features2) return 0;
    return obj->impl->CalculateSimilarity(features1, features2);
}

void SeetaFaceRecognizer_SetSingleCalculationThreads(int num)
{
    seeta::FaceRecognizer::SetSingleCalculationThreads(num);
}
