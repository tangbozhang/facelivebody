#ifndef HOLIDAY_REAL_MUL_GPU_H_
#define HOLIDAY_REAL_MUL_GPU_H_

#include "HolidayBaseLayer.h"
#include "HolidayNetResource.h"
#include "HolidayCommonCuda.h"

class HolidayRealMulGPU : public HolidayBaseLayer<float>
{
public:
    HolidayRealMulGPU();
    ~HolidayRealMulGPU();
	int Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map);
	int Init(Holiday_LayerParameter &inputparam, HolidayNetResource<float> *pNetResource);
	
	int Exit();
private:
    HolidayNetResourceGpu *pNetResourceGpu;

    std::vector<int> m_y_shape;
    std::shared_ptr<float> m_y_data;
	
	float *pf_y_gpu;
};



#endif // !HOLIDAY_REAL_MUL_GPU_H_

