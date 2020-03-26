#ifndef _HOLIDAY_SPLIT_GPU_H__
#define _HOLIDAY_SPLIT_GPU_H__

#include"HolidayBaseLayer.h"
#include "HolidayCommonCuda.h"

class HolidaySplitGPU : public HolidayBaseLayer<float>
{
public:
	HolidaySplitGPU();
	~HolidaySplitGPU();
	
	int Exit();
	int Init(Holiday_LayerParameter& inputparam, HolidayNetResource<float> *pNetResource);
	int Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map);
	
private:
	HolidayNetResourceGpu *pNetResourceGpu;
};

#endif