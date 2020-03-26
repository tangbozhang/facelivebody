#include "HolidayNetResource.h"
#include "HolidayBaseLayer.h"
#include "HolidayCommonCuda.h"

class HolidayReluGPU : public HolidayBaseLayer<float>
{
public:
	HolidayReluGPU();
	~HolidayReluGPU();
	int Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map);
	int Init(Holiday_LayerParameter& inputparam, HolidayNetResource<float> *pNetResource);
	int Exit();
private:
	HolidayNetResourceGpu  *pNetResourceGpu;
	float fNegative_slope;
    bool m_has_max = false;
    float m_max;
};