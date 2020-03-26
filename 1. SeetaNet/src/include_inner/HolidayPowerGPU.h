#ifndef _HOLIDAY_POWER_GPU_H__
#define _HOLIDAY_POWER_GPU_H__


#include "HolidayNetResource.h"
#include "HolidayBaseLayer.h"
#include "HolidayCommonCuda.h"

class HolidayPowerGPU : public HolidayBaseLayer<float>
{
public:
	HolidayPowerGPU();
	~HolidayPowerGPU();
	int Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map);
	int Init(Holiday_LayerParameter& inputparam, HolidayNetResource<float> *pNetResource);
	int Exit();
private:
	HolidayNetResourceGpu *pNetResourceGpu;
	float fScale;
	float fShift;
	float fPower;
};


#endif