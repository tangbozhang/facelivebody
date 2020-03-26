#include "VIPLCFaceDetector.h"

#include "VIPLFaceDetector.h"

VIPLCFaceDetector* VIPLCNewFaceDetector(const char* model_path, VIPLCDevice device)
{
	try
	{
		return reinterpret_cast<VIPLCFaceDetector *>(
			new VIPLFaceDetector(model_path, VIPLFaceDetector::Device(device)));
	}
	catch(...)
	{
		return nullptr;
	}
}

void VIPLCDeleteFaceDetector(const VIPLCFaceDetector* obj)
{
	if (obj == nullptr) return;
	auto impl = reinterpret_cast<const VIPLFaceDetector *>(obj);
	delete impl;
}

template <typename T>
static T *NewArray(const std::vector<T> &list)
{
	if (list.empty()) return nullptr;
	T *data = new T[list.size()];
	for (size_t i = 0; i < list.size(); ++i) data[i] = list[i];
	return data;
}

VIPLCFaceInfoArray VIPLCFaceDetector_Detect(VIPLCFaceDetector* obj, VIPLImageData image)
{
	if (obj == nullptr) return {nullptr, 0};
	auto impl = reinterpret_cast<VIPLFaceDetector *>(obj);
	auto infos = impl->Detect(image);

	struct VIPLCFaceInfoArray arr;
	arr.size = int(infos.size());
	arr.data = NewArray(infos);

	return arr;
}

void VIPLCDeleteFaceInfoArray(const VIPLCFaceInfoArray obj)
{
	delete[] obj.data;
}

void VIPLCFaceDetector_SetMinFaceSize(VIPLCFaceDetector* obj, int32_t size)
{
	if (obj == nullptr) return;
	auto impl = reinterpret_cast<VIPLFaceDetector *>(obj);

	impl->SetMinFaceSize(size);
}

void VIPLCFaceDetector_SetImagePyramidScaleFactor(VIPLCFaceDetector* obj, float factor)
{
	if (obj == nullptr) return;
	auto impl = reinterpret_cast<VIPLFaceDetector *>(obj);

	impl->SetImagePyramidScaleFactor(factor);
}

void VIPLCFaceDetector_SetScoreThresh(VIPLCFaceDetector* obj, float thresh1, float thresh2, float thresh3)
{
	if (obj == nullptr) return;
	auto impl = reinterpret_cast<VIPLFaceDetector *>(obj);

	impl->SetScoreThresh(thresh1, thresh2, thresh3);
}

void VIPLCFaceDetector_SetVideoStable(VIPLCFaceDetector* obj, int stable)
{
	if (obj == nullptr) return;
	auto impl = reinterpret_cast<VIPLFaceDetector *>(obj);

	impl->SetVideoStable(stable != 0);
}

int VIPLCFaceDetector_GetVideoStable(VIPLCFaceDetector* obj)
{
	if (obj == nullptr) return 0;
	auto impl = reinterpret_cast<VIPLFaceDetector *>(obj);

	return impl->GetVideoStable();
}

void VIPLCFaceDetector_SetNumThreads(int num)
{
	VIPLFaceDetector::SetNumThreads(num);
}

int32_t VIPLCFaceDetector_GetMinFaceSize(VIPLCFaceDetector* obj)
{
	if (obj == nullptr) return 0;
	auto impl = reinterpret_cast<VIPLFaceDetector *>(obj);

	return impl->GetMinFaceSize();
}

float VIPLCFaceDetector_GetImagePyramidScaleFactor(VIPLCFaceDetector* obj)
{
	if (obj == nullptr) return 0;
	auto impl = reinterpret_cast<VIPLFaceDetector *>(obj);

	return impl->GetImagePyramidScaleFactor();
}

void VIPLCFaceDetector_GetScoreThresh(VIPLCFaceDetector* obj, float* thresh1, float* thresh2, float* thresh3)
{
	if (obj == nullptr) return;
	auto impl = reinterpret_cast<VIPLFaceDetector *>(obj);

	impl->GetScoreThresh(thresh1, thresh2, thresh3);
}

VIPLCFaceDetector_CoreSize VIPLCFaceDetector_GetCoreSize(VIPLCFaceDetector* obj)
{
	VIPLCFaceDetector_CoreSize core_size = {0, 0};

	if (obj == nullptr) return core_size;
	auto impl = reinterpret_cast<VIPLFaceDetector *>(obj);

	auto size = impl->GetCoreSize();
	core_size.width = size.width;
	core_size.height = size.height;

	return core_size;
}
