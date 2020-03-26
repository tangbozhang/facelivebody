#ifndef HOLIDAY_CONVOLUTION_GPU_H
#define HOLIDAY_CONVOLUTION_GPU_H

#include "HolidayBaseLayer.h"
//#include "HolidayCNN_proto.pb.h"
//#include "HolidayCommon.h"
//#include "HolidayFeatureMap.h"
//#include "HolidayBlobGpu.h"
//#include "MacroHoliday.h"
#include "HolidayCommonCuda.h"

class HolidayConvolutionGPU : public HolidayBaseLayer<float>
{
public:
	HolidayConvolutionGPU();
	~HolidayConvolutionGPU();
	int Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map);
	int Init(Holiday_LayerParameter &inputparam, HolidayNetResource<float> *pdjNetResource);
	int Exit();

	int Caculate(const int height, const int width,
				 const int kernel_h, const int kernel_w,
				 const int pad_h, const int pad_w,
				 const int stride_h, const int stride_w,
				 const int dilation_h, const int dilation_w,
				 int& output_h, int& output_w);

	int Calculate(const std::vector<int> &bottom_shape, std::vector<int> &top_shape);

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
	int dwKernelNum;
	int dwKernelRows;
	int dwKernelCols;
	int dwKernelSlices;
	int dwGroup;
	float *pfKernel_d;
	float **ppfKernel_d;
	float *pfBias_d;

	std::string m_tf_padding;
	int m_tf_fake_padding_h = 0;
    int m_tf_fake_padding_w = 0;
    int m_tf_conv_shift_h = 0;
    int m_tf_conv_shift_w = 0;

};

#endif