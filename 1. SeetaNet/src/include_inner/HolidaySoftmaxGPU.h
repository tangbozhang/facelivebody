#include "HolidayNetResource.h"
#include "HolidayBaseLayer.h"
#include "HolidayCommonCuda.h"

class HolidaySoftMaxGPU : public HolidayBaseLayer<float>
{
public:
	HolidaySoftMaxGPU();
	~HolidaySoftMaxGPU();
	int Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map);
	int Init(Holiday_LayerParameter& inputparam, HolidayNetResource<float> *pNetResource);
	int Exit();
private:
	HolidayNetResourceGpu *pNetResourceGpu;
	float *pfMax_d;
	float *pfSum_d;
};