#include "HolidayCommonCuda.h"
#include "HolidayBaseLayer.h"
class HolidayInnerProductGPU : public HolidayBaseLayer<float>
{
public:
	HolidayInnerProductGPU();
	~HolidayInnerProductGPU();
	int Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map);
	int Init(Holiday_LayerParameter& inputparam, HolidayNetResource<float> *pNetResource);
	int Exit();
private:
	HolidayNetResourceGpu *pNetResourceGpu;
	float *pfParam_d;
	float *pfBias_d;
};
