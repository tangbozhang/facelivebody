#include "seeta/CFaceDetector.h"
#include "seeta/FaceDetector.h"

struct SeetaFaceDetector
{
	seeta::FaceDetector *impl;
};

SeetaFaceDetector* SeetaNewFaceDetector(SeetaModelSetting setting)
{
    std::unique_ptr<SeetaFaceDetector> obj(new SeetaFaceDetector);
    try
    {
        obj->impl = new seeta::FaceDetector(setting);
    }
    catch (const std::exception &)
    {
        return nullptr;
    }
    return obj.release();
}

SeetaFaceDetector* SeetaNewFaceDetectorWithCoreSize(SeetaModelSetting setting, const SeetaSize core_size)
{
    std::unique_ptr<SeetaFaceDetector> obj(new SeetaFaceDetector);
    try
    {
        obj->impl = new seeta::FaceDetector(setting, core_size);
    }
    catch (const std::exception &)
    {
        return nullptr;
    }
    return obj.release();
}

void SeetaDeleteFaceDetector(const SeetaFaceDetector* obj)
{
	if (!obj) return;
	delete obj->impl;
	delete obj;
}

SeetaFaceInfoArray SeetaFaceDetector_Detect(SeetaFaceDetector* obj, const SeetaImageData image)
{
	static const SeetaFaceInfoArray result = {nullptr, 0};
	if (!obj || !obj->impl) return result;
	return obj->impl->Detect(image);
}

void SeetaFaceDetector_SetMinFaceSize(SeetaFaceDetector* obj, int size)
{
	if (!obj || !obj->impl) return;
	obj->impl->SetMinFaceSize(size);
}

void SeetaFaceDetector_SetImagePyramidScaleFactor(SeetaFaceDetector* obj, float factor)
{
	if (!obj || !obj->impl) return;
	obj->impl->SetImagePyramidScaleFactor(factor);
}

void SeetaFaceDetector_SetScoreThresh(SeetaFaceDetector* obj, float thresh1, float thresh2, float thresh3)
{
	if (!obj || !obj->impl) return;
	obj->impl->SetScoreThresh(thresh1, thresh2, thresh3);
}

void SeetaFaceDetector_SetVideoStable(SeetaFaceDetector* obj, int stable)
{
	if (!obj || !obj->impl) return;
	obj->impl->SetVideoStable(stable != 0);
}

int SeetaFaceDetector_GetVideoStable(SeetaFaceDetector* obj)
{
	if (!obj || !obj->impl) return 0;
	return obj->impl->GetVideoStable();
}

int SeetaFaceDetector_GetMinFaceSize(SeetaFaceDetector* obj)
{
    if (!obj || !obj->impl) return 0;
    return obj->impl->GetMinFaceSize();
}

float SeetaFaceDetector_GetImagePyramidScaleFactor(SeetaFaceDetector* obj)
{
    if (!obj || !obj->impl) return 0;
    return obj->impl->GetImagePyramidScaleFactor();
}

void SeetaFaceDetector_GetScoreThresh(SeetaFaceDetector* obj, float* thresh1, float* thresh2, float* thresh3)
{
    if (!obj || !obj->impl) return;
    obj->impl->GetScoreThresh(thresh1, thresh2, thresh3);
}

int SeetaFaceDetector_SetLogLevel(int level)
{
    return seeta::FaceDetector::SetLogLevel(level);
}

void SeetaFaceDetector_SetSingleCalculationThreads(int num)
{
    seeta::FaceDetector::SetSingleCalculationThreads(num);
}
