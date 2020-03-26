#ifndef _HOLIDAY_ELTWISE_GPU_H__
#define _HOLIDAY_ELTWISE_GPU_H__

#include "HolidayBaseLayer.h"
#include "HolidayCommonCuda.h"

class HolidayEltwiseGPU : public HolidayBaseLayer<float>
{
public:
	HolidayEltwiseGPU();
	~HolidayEltwiseGPU();
	int Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map);
	int Init(Holiday_LayerParameter& inputparam, HolidayNetResource<float> *pNetResource);
	
	int Exit();
private:
	float **ppfCurDataIn;
	float **ppfCurDataIn_d;
	HolidayNetResourceGpu *pNetResourceGpu;
	float *pfCoeff_d;
	int dwType;
	std::vector<float> eltwise_coeff;
};

#endif
