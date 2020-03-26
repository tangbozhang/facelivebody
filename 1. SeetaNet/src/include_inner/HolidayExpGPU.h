#ifndef _HOLIDAY_EXP_GPU_H_
#define _HOLIDAY_EXP_GPU_H_

#include "HolidayNetResource.h"
#include "HolidayBaseLayer.h"
#include "HolidayCommonCuda.h"

class HolidayExpGPU : public HolidayBaseLayer<float>
{
public:
	HolidayExpGPU();
	~HolidayExpGPU();
	int Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map);
	int  Init(Holiday_LayerParameter& inputparam, HolidayNetResource<float> *pNetResource);

	int Exit();
private:
	HolidayNetResourceGpu *pNetResourceGpu;
	float fScaleIn;
	float fScaleOut;
};

#endif