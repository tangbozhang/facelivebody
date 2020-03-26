#ifndef __HOLIDAY_DECONVOLUTION_GPU_H_
#define __HOLIDAY_DECONVOLUTION_GPU_H_

#include "HolidayBaseLayer.h"
#include "HolidayCommonCuda.h"


class HolidayDeconvolutionGPU : public HolidayBaseLayer<float>
{
public:
    HolidayDeconvolutionGPU();
    ~HolidayDeconvolutionGPU();
    int Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map);
    int Init(Holiday_LayerParameter &inputparam, HolidayNetResource<float> *pdjNetResource);
    int Exit();
private:
    HolidayNetResourceGpu *pNetResourceGpu;
    float **ppfBlas;
    float **ppfBlas_d;
    int dwStrideH;
    int dwStrideW;
    int dwPadH;
    int dwPadW;
    int dwDilationH;
    int dwDilationW;
    int dwKernelRows;
    int dwKernelCols;
    int dwKernelNum;
    int dwGroup;
    int dwWeightRows;
    int dwWeightCols;
    int dwKernelSlices;
    
    float *pfWeight_d;
    float **ppfWeight_d;
    float *pfBias_d;

    float *bias_multiplier_;
};


#endif // !__HOLIDAY_DECONVOLUTION_GPU_H__