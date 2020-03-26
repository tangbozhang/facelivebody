#ifndef _MEMORY_DATA_LAYER_GPU_H__
#define _MEMORY_DATA_LAYER_GPU_H__

#include "HolidayBaseLayer.h"
#include "HolidayCommonCuda.h"

class HolidayMemoryDataLayerGPU :public HolidayBaseLayer<float>
{
public:
	HolidayMemoryDataLayerGPU();
	~HolidayMemoryDataLayerGPU();

	int Init(Holiday_LayerParameter& inputparam, HolidayNetResource<float> *pNetResource);
	int Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map);
	int Exit();

public:

	float data_scale;
	int32_t m_mean_type;

	

private:
	float fScale;
	float *pfMean_d;
	HolidayNetResourceGpu  *pNetResourceGpu;
};


#endif