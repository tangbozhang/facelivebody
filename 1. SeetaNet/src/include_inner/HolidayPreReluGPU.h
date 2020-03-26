#include "HolidayBaseLayer.h"
#include "HolidayCommonCuda.h"

class HolidayPreReluGPU : public HolidayBaseLayer<float>
{
public:
	HolidayPreReluGPU();
	~HolidayPreReluGPU();
	int Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map);
	int Init(Holiday_LayerParameter& inputparam, HolidayNetResource<float> *pNetResource);
	int Exit();
private:
	HolidayNetResourceGpu *pNetResourceGpu;
	float *pfSlope_d;
};
