#ifndef _HOLIDAY_FEATURE_MAP_H__
#define _HOLIDAY_FEATURE_MAP_H__

#include "HolidayCommon.h"
#include "HolidayNetResource.h"
#include"HolidayBlobCpu.h"

#ifdef HOLIDAY_GPU
#include "HolidayBlobGpu.h"
#endif



enum DATA_STORAGE_TYPE
{
	DATA_INVALID = 0,
	DATA_CPU_WIDTH = 1,
	DATA_CPU_SLICE = 2,
	DATA_CPU_WIDTH_CHAR = 3,
	DATA_CPU_SLICE_CHAR = 4,
	DATA_GPU = 5,
	DATA_OPENCL = 6
};
template<typename T>
class HolidayFeatureMap
{
public:

	HolidayFeatureMap(){};
	~HolidayFeatureMap(){};

	int TransFormDataIn();

	std::string data_name;

	//HolidayDataSize data_shape;
	std::vector<int> data_shape;
	
	int dwStorageType;
	HolidayNetResource<T>* pNetResource;
	HolidayBlobCpu<T> m_cpu;
#ifdef HOLIDAY_GPU
    HolidayBlobGpu m_gpu;

	T *gpu_ptr() { return m_gpu.pfData_gpu; }
#endif

	std::vector<int> &shape() { return data_shape; }

	const std::vector<int> &shape() const { return data_shape; }

	int &shape(size_t axis) { return data_shape[axis]; }

	const int &shape(size_t axis) const { return data_shape[axis]; }

	int count() const
	{
		int mul = 1;
		for (auto dim : data_shape) mul *= dim;
		return mul;
	}

	T *cpu_ptr() { return m_cpu.dataMemoryPtr(); }
};

template<typename T>
int HolidayFeatureMap<T>::TransFormDataIn()
{
	switch (dwStorageType)
	{
	case DATA_CPU_WIDTH:
	{
		break;
	}
	case DATA_GPU:
	{
		break;
	}
	default:
		break;
	}
	return 0;
}

template<>
inline int HolidayFeatureMap<float>::TransFormDataIn()
{
	switch (dwStorageType)
	{
	case DATA_CPU_WIDTH:
	{
		break;
	}
	case DATA_GPU:
	{
#ifdef HOLIDAY_GPU
		//m_data_gpu.Gpu_DataOut(pdjNetResource->pNetResourceGpu, DATA_CPU_SLICE, m_data2.dataMemoryPtr());
		m_gpu.Gpu_DataOut(pNetResource->pNetResourceGpu, DATA_CPU_WIDTH, cpu_ptr());
#endif
		break;
	}
	default:
		break;
	}
	return 0;
}


#endif