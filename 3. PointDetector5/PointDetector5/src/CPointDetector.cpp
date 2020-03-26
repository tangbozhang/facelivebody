#include "seeta/CPointDetector.h"
#include "seeta/PointDetector.h"

struct SeetaPointDetector
{
	seeta::PointDetector *impl;
};

SeetaPointDetector* SeetaNewPointDetector(SeetaModelSetting setting)
{
	std::unique_ptr<SeetaPointDetector> obj(new SeetaPointDetector);
	obj->impl = new seeta::PointDetector(setting);
	return obj.release();
}

void SeetaDeletePointDetector(const SeetaPointDetector* obj)
{
	if (!obj) return;
	delete obj->impl;
	delete obj;
}

int SeetaPointDetector_GetLandmarkNumber(const SeetaPointDetector* obj)
{
	if (!obj || !obj->impl) return 0;
	return obj->impl->GetLandmarkNumber();
}

int SeetaPointDetector_Detect(const SeetaPointDetector* obj, const SeetaImageData image, const SeetaRect face, SeetaPointF* points)
{
	if (!obj || !obj->impl) return 0;
	if (!points) return 0;
	return int(obj->impl->Detect(image, face, points));
}

int SeetaPointDetector_SetLogLevel(int level)
{
    return seeta::PointDetector::SetLogLevel(level);
}

int SeetaPointDetector_DetectMasks(const SeetaPointDetector* obj, const SeetaImageData image, const SeetaRect face,
    SeetaPointF* points, int* masks)
{
    if (!obj || !obj->impl) return 0;
    if (!points) return 0;
    return int(obj->impl->Detect(image, face, points, masks));
}
