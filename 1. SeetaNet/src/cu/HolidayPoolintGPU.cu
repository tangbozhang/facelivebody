#include "HolidayPoolintGPU.h"
#include "HolidayCommonCuda.h"

enum HolidayPoolingParameter_PoolMethod {
	PoolingParameter_PoolMethod_MAX = 0,
	PoolingParameter_PoolMethod_AVE = 1,
	PoolingParameter_PoolMethod_STOCHASTIC = 2
};
Pooling_gpu::Pooling_gpu()
{
}
Pooling_gpu::~Pooling_gpu()
{
}
__global__ static void gMaxPooling_kernel(float  *pfDataIn, float *pfDataOut, int dwSize,
	int dwRowIn, int dwColIn, int dwSliceIn, int dwRowOut, int dwColOut, int dwSliceOut,
	int dwKernelH, int dwKernelW, int dwStrideH, int dwStrideW, int dwPadH, int dwPadW)
{
	int dwIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (dwIdx < dwSize)
	{
		int dwDimN = dwIdx / (dwSliceOut * dwRowOut * dwColOut);
		int dwDim2S = dwIdx % (dwSliceOut * dwRowOut * dwColOut) / (dwRowOut * dwColOut);
		int dwDim2R = dwIdx % (dwRowOut * dwColOut) / dwColOut;
		int dwDim2C = dwIdx % dwColOut;
		int dwDim1R = dwDim2R * dwStrideH - dwPadH;
		int dwDim1C = dwDim2C * dwStrideW - dwPadW;
		int dwDim1S = dwDim2S;
		float fMax;
		fMax = -3.402823E38;
		for (int i = Holliday_MAX(dwDim1R, 0); i < Holliday_MIN(dwDim1R + dwKernelH, dwRowIn); i++)
		{
			for (int j = Holliday_MAX(dwDim1C, 0); j < Holliday_MIN(dwDim1C + dwKernelW, dwColIn); ++j)
			{
				fMax = fmaxf(fMax, pfDataIn[j + dwColIn * (i + dwRowIn * (dwDim1S + dwSliceIn * dwDimN))]);
			}
		}
		pfDataOut[dwIdx] = fMax;
	}
}
__global__ static void gAveragePooling_kernel(float  *pfDataIn, float *pfDataOut, int dwSize,
	int dwRowIn, int dwColIn, int dwSliceIn, int dwRowOut, int dwColOut, int dwSliceOut,
	int dwKernelH, int dwKernelW, int dwStrideH, int dwStrideW, int dwPadH, int dwPadW)
{
	int dwIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (dwIdx < dwSize)
	{
		int dwDimN = dwIdx / (dwSliceOut * dwRowOut * dwColOut);
		int dwDim2S = dwIdx % (dwSliceOut * dwRowOut * dwColOut) / (dwRowOut * dwColOut);
		int dwDim2R = dwIdx % (dwRowOut * dwColOut) / dwColOut;
		int dwDim2C = dwIdx % dwColOut;
		int dwDim1R = dwDim2R * dwStrideH - dwPadH;
		int dwDim1C = dwDim2C * dwStrideW - dwPadW;
		int dwDim1S = dwDim2S;
		float fSum;
		int dwCount;
		fSum = 0.f;
		dwCount = 0;
		for (int i = Holliday_MAX(dwDim1R, 0); i < Holliday_MIN(dwDim1R + dwKernelH, dwRowIn); i++)
		{
			for (int j = Holliday_MAX(dwDim1C, 0); j < Holliday_MIN(dwDim1C + dwKernelW, dwColIn); ++j)
			{
				fSum += pfDataIn[j + dwColIn * (i + dwRowIn * (dwDim1S + dwSliceIn * dwDimN))];
				++dwCount;
			}
		}
		pfDataOut[dwIdx] = fSum / dwCount;
	}
}
int Pooling_gpu::Init(Holiday_LayerParameter& inputparam, HolidayNetResource<float> *pNetResource)
{
	pNetResourceGpu = (HolidayNetResourceGpu *)pNetResource->pNetResourceGpu;

	dwType = inputparam.pooling_param().pool();
	
	int index = inputparam.bottom_index(0);
	bottom_data_size.resize(1);
	bottom_data_size[0] = pNetResource->feature_vector_size[index];
	
	
	dwStrideH = inputparam.pooling_param().stride_height();
	dwStrideW = inputparam.pooling_param().stride_width();
	dwKernelH = inputparam.pooling_param().kernel_height();
	dwKernelW = inputparam.pooling_param().kernel_width();
	dwPadH = inputparam.pooling_param().pad_height();
	dwPadW = inputparam.pooling_param().pad_width();

	dwValid = false;
	if (inputparam.pooling_param().has_valid())
	{
		dwValid = inputparam.pooling_param().valid();
	}

	if (inputparam.pooling_param().global_pooling())
	{
		dwKernelH = bottom_data_size[0].data_dim[2];
		dwKernelW = bottom_data_size[0].data_dim[3];
		dwPadH = 0;
		dwPadW = 0;
	}
	top_data_size.resize(1);
	top_data_size[0].data_dim[1] = bottom_data_size[0].data_dim[1];
	if (dwValid)
	{
		top_data_size[0].data_dim[2] = floor((bottom_data_size[0].data_dim[2] - dwKernelH + 2 * dwPadH) / (float)dwStrideH + 1);
		top_data_size[0].data_dim[3] = floor((bottom_data_size[0].data_dim[3] - dwKernelW + 2 * dwPadW) / (float)dwStrideW + 1);
	}
	else
	{
		top_data_size[0].data_dim[2] = ceil((bottom_data_size[0].data_dim[2] - dwKernelH + 2 * dwPadH) / (float)dwStrideH + 1);
		top_data_size[0].data_dim[3] = ceil((bottom_data_size[0].data_dim[3] - dwKernelW + 2 * dwPadW) / (float)dwStrideW + 1);
	}
	//int dwSize = top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3] * top_data_size[0].data_dim[1];

	return CUDA_RETURN_VALUE;
}
int Pooling_gpu::Exit()
{
	return CUDA_RETURN_VALUE;
}
int Pooling_gpu::GetTopSize(std::vector<HolidayDataSize>& out_data_size)
{
	out_data_size = top_data_size;
	return CUDA_RETURN_VALUE;
}
int Pooling_gpu::Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map)
{
#ifdef _DEBUG
	cudaEvent_t start1;
	cudaEventCreate(&start1);
	cudaEvent_t stop1;
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, NULL);
#endif
	input_data_map[0]->m_gpu.Gpu_DataIn(pNetResourceGpu, input_data_map[0]->dwStorageType, input_data_map[0]->m_cpu.dataMemoryPtr());
	if (PoolingParameter_PoolMethod_AVE== dwType)
	{
		int dwSize = top_data_size[0].data_dim[1] * top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3];
		gAveragePooling_kernel << <CUDA_BLOCK(dwSize * input_data_map[0]->m_gpu.shape_[0], CUDA_THREAD_NUM), CUDA_THREAD_NUM, 0, pNetResourceGpu->main_stream>> >
			(input_data_map[0]->m_gpu.pfData_gpu, output_data_map[0]->m_gpu.pfData_gpu,
		dwSize* input_data_map[0]->m_gpu.shape_[0],
		bottom_data_size[0].data_dim[2], bottom_data_size[0].data_dim[3], bottom_data_size[0].data_dim[1],
		top_data_size[0].data_dim[2], top_data_size[0].data_dim[3], top_data_size[0].data_dim[1],
		dwKernelH, dwKernelW, dwStrideH, dwStrideW, dwPadH, dwPadW);
	}
	else if (PoolingParameter_PoolMethod_MAX == dwType)
	{
		int dwSize = top_data_size[0].data_dim[1] * top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3];
		gMaxPooling_kernel << <CUDA_BLOCK(dwSize * input_data_map[0]->m_gpu.shape_[0], CUDA_THREAD_NUM), CUDA_THREAD_NUM, 0, pNetResourceGpu->main_stream>> >
			(input_data_map[0]->m_gpu.pfData_gpu, output_data_map[0]->m_gpu.pfData_gpu,
			dwSize * input_data_map[0]->m_gpu.shape_[0],
			bottom_data_size[0].data_dim[2], bottom_data_size[0].data_dim[3], bottom_data_size[0].data_dim[1],
			top_data_size[0].data_dim[2], top_data_size[0].data_dim[3], top_data_size[0].data_dim[1],
			dwKernelH, dwKernelW, dwStrideH, dwStrideW, dwPadH, dwPadW);
	}
	else if (PoolingParameter_PoolMethod_STOCHASTIC == dwType)
	{
	}
	else
	{
	}
	output_data_map[0]->dwStorageType = DATA_GPU;
	output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];
	output_data_map[0]->data_shape[1] = top_data_size[0].data_dim[1];
	output_data_map[0]->data_shape[2] = top_data_size[0].data_dim[2];
	output_data_map[0]->data_shape[3] = top_data_size[0].data_dim[3];
	return CUDA_RETURN_VALUE;
}