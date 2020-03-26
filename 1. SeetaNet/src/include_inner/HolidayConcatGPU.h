#ifndef _HOLIDAY_CONCAT_GPU__
#define _HOLIDAY_CONCAT_GPU__
#include "HolidayBaseLayer.h"
#include "HolidayCommonCuda.h"

class HolidayConcatGPU : public HolidayBaseLayer<float>
{
public:
	HolidayConcatGPU();
	~HolidayConcatGPU();
	int Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map);
	int Init(Holiday_LayerParameter& inputparam, HolidayNetResource<float> *pNetResource);
	int Exit();
private:
	HolidayNetResourceGpu *pNetResourceGpu;
	int *pdwWidthIn;
	int dwWidthOut;

	int dwConcatAxis;
};



#endif
