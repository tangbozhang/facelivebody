#include "VIPLCPointDetector.h"

#include "VIPLPointDetector.h"

using VIPLCClass = VIPLPointDetector;

VIPLCPointDetector* VIPLCNewPointDetector(const char* model_path)
{
	try
	{
		return reinterpret_cast<VIPLCPointDetector *>(
			new VIPLCClass(model_path));
	}
	catch (...)
	{
		return nullptr;
	}
}

void VIPLCDeletePointDetector(const VIPLCPointDetector* obj)
{
	if (obj == nullptr) return;
	auto impl = reinterpret_cast<const VIPLCClass *>(obj);
	delete impl;
}

void VIPLCPointDetector_LoadModel(struct VIPLCPointDetector *obj, const char* model_path)
{
	if (obj == nullptr) return;
	auto impl = reinterpret_cast<VIPLCClass *>(obj);

	impl->LoadModel(model_path);
}

void VIPLCPointDetector_SetStable(VIPLCPointDetector* obj, int is_stable)
{
	if (obj == nullptr) return;
	auto impl = reinterpret_cast<VIPLCClass *>(obj);

    impl->SetStable(is_stable != 0);
}

int VIPLCPointDetector_LandmarkNum(const VIPLCPointDetector* obj)
{
	if (obj == nullptr) return 0;
	auto impl = reinterpret_cast<const VIPLCClass *>(obj);

	return impl->LandmarkNum();
}

int VIPLCPointDetector_DetectCroppedLandmarks(const VIPLCPointDetector* obj, VIPLImageData cropped_image, VIPLPoint* landmarks)
{
	if (obj == nullptr) return 0;
	auto impl = reinterpret_cast<const VIPLCClass *>(obj);

	return impl->DetectCroppedLandmarks(cropped_image, landmarks);
}

int VIPLCPointDetector_DetectLandmarks(const VIPLCPointDetector* obj, VIPLImageData image, VIPLFaceInfo info, VIPLPoint* landmarks)
{
	if (obj == nullptr) return 0;
	auto impl = reinterpret_cast<const VIPLCClass *>(obj);

	return impl->DetectLandmarks(image, info, landmarks);
}
