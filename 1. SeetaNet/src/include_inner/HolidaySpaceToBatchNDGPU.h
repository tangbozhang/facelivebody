#ifndef __HOLIDAY_SPACE_TO_BATCH_ND_GPU_H_
#define __HOLIDAY_SPACE_TO_BATCH_ND_GPU_H_

#include "HolidayBaseLayer.h"
#include "HolidayNetResource.h"
#include "HolidayCommonCuda.h"

class HolidaySpaceToBatchNDGPU : public HolidayBaseLayer<float>
{
public:
    HolidaySpaceToBatchNDGPU();
	~HolidaySpaceToBatchNDGPU();

	int Init(Holiday_LayerParameter& inputparam, HolidayNetResource<float> *pdjNetResource) override;

    int Exit() override;
	
	int Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map) override;

    void CaculateOutputSize(int input_number, int input_height, int input_width, int input_channels,
                            int &output_number, int &output_height, int &output_width, int &output_channels);

    void CaculateOutputSize(const std::vector<int> &input_shape, std::vector<int> &output_shape) {
        output_shape.resize(4);
        CaculateOutputSize(input_shape[0], input_shape[2], input_shape[3], input_shape[1],
                           output_shape[0], output_shape[2], output_shape[3], output_shape[1]);
    }
private:
    HolidayNetResourceGpu *pNetResourceGpu;
public:
    std::vector<int> m_block_shape;
    std::vector<int> m_paddings;
    int *m_block_shape_gpu = nullptr;
    int *m_paddings_gpu = nullptr;
};

#endif