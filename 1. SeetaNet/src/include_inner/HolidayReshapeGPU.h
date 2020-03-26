#pragma once

#include "HolidayNetResource.h"
#include "HolidayBaseLayer.h"
#include "HolidayCommonCuda.h"

class HolidayReshapeGPU : public HolidayBaseLayer<float>
{
public:
    HolidayReshapeGPU();
    ~HolidayReshapeGPU();
	int Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map);
	int Init(Holiday_LayerParameter& inputparam, HolidayNetResource<float> *pNetResource);
	int Exit();
private:
	HolidayNetResourceGpu  *pNetResourceGpu;
	std::vector<int> m_shape;
    std::vector<int> m_permute;
};