#ifndef HOLIDAYSACLE_GPU_H_
#define HOLIDAYSACLE_GPU_H_

#include "HolidayBaseLayer.h"
#include "HolidayNetResource.h"
#include "HolidayCommonCuda.h"

class HolidayScaleGPU : public HolidayBaseLayer<float>
{
public:
	HolidayScaleGPU();
	~HolidayScaleGPU();
	int Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map);
	int Init(Holiday_LayerParameter &inputparam, HolidayNetResource<float> *pNetResource);
	
	int Exit();
private:
	HolidayNetResourceGpu *pNetResourceGpu;
	
	float *pfScale_d;
	float *pfBias_d;
};



#endif // !HOLIDAYSACLE_GPU_H_

