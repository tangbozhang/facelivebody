#ifndef _HOLIDAY_LRN_GPU_H__
#define _HOLIDAY_LRN_GPU_H__

#include"HolidayBaseLayer.h"
#include "HolidayCommonCuda.h"
class HolidayLocalResponseNormalizeGPU : public HolidayBaseLayer<float>
{
public:
	HolidayLocalResponseNormalizeGPU();
	~HolidayLocalResponseNormalizeGPU();

	int Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map);
	int Init(Holiday_LayerParameter& inputparam, HolidayNetResource<float> *pNetResource);
	int GetTopSize(std::vector<HolidayDataSize>& out_data_size);
	int Exit();
private:
	HolidayNetResourceGpu *pNetResourceGpu;
	float alpha_d;
	float beta_d;
	int localsize_d;
	int k_d;
	int normalize_type;
};

#endif