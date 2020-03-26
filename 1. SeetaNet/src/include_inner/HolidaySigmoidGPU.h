#ifndef _HOLIDAY_SIGMOID_GPU_H__
#define _HOLIDAY_SIGMOID_GPU_H__

#include "HolidayCommonCuda.h"
#include "HolidayBaseLayer.h"

class HolidaySigmoidGPU : public HolidayBaseLayer<float>
{
public:
	HolidaySigmoidGPU();
	~HolidaySigmoidGPU();
	int Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map);
	int Init(Holiday_LayerParameter& inputparam, HolidayNetResource<float> *pNetResource);
	int Exit();
private:
	HolidayNetResourceGpu *pNetResourceGpu;
};
#endif