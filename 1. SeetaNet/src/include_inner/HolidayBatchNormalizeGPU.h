#ifndef _HOLIDAY_BATCH_NORMALISE_GPU_H__
#define _HOLIDAY_BATCH_NORMALISE_GPU_H__
#include "HolidayBaseLayer.h"
#include "HolidayNetResource.h"
#include "HolidayCommonCuda.h"

//template<class T>
class HolidayBatchNormalizeGPU : public HolidayBaseLayer<float>
{
public:
	HolidayBatchNormalizeGPU();
	~HolidayBatchNormalizeGPU();
	int Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>&output_data_map);
	int Init(Holiday_LayerParameter& inputparam, HolidayNetResource<float> *pNetResource);
	int Exit();
private:
	HolidayNetResourceGpu *pNetResourceGpu;
	float *pfMean_d;
	float *pfVariance_d;
};



#endif