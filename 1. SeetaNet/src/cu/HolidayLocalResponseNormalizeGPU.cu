#include"HolidayLocalResponseNormalizeGPU.h"

HolidayLocalResponseNormalizeGPU::HolidayLocalResponseNormalizeGPU()
{
}
HolidayLocalResponseNormalizeGPU::~HolidayLocalResponseNormalizeGPU()
{
}
__global__ static void gLocalResponseNormalize_kernel(float *pfDataIn, float *pfDataOut, int dwSize,
	int dwRow, int dwCol, int dwSlice, float fAlpha, float fBeta, float fK, int dwLocalSize)
{
	int dwIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (dwIdx < dwSize)
	{
		int dwDimN = dwIdx / (dwSlice * dwRow * dwCol);
		int dwDimS = dwIdx % (dwSlice * dwRow * dwCol) / (dwRow * dwCol);
		int dwDimR = dwIdx % (dwRow * dwCol) / dwCol;
		int dwDimC = dwIdx % dwCol;
		float fSum = 0.f;
		for (int i = Holliday_MAX(0, dwDimS - (dwLocalSize >> 1)); i <= Holliday_MIN(dwSlice - 1, dwDimS + (dwLocalSize >> 1)); i++)
		{
			fSum += pfDataIn[dwDimC + dwCol * (dwDimR + dwRow * (i + dwSlice * dwDimN))]
				* pfDataIn[dwDimC + dwCol * (dwDimR + dwRow * (i + dwSlice * dwDimN))];
		}
		pfDataOut[dwIdx] = pfDataIn[dwIdx] * powf(fAlpha * fSum + fK, -fBeta);
	}
}

int HolidayLocalResponseNormalizeGPU::Init(Holiday_LayerParameter& inputparam, HolidayNetResource<float> *pNetResource)
{
	pNetResourceGpu = (HolidayNetResourceGpu *)pNetResource->pNetResourceGpu;


	int index = inputparam.bottom_index(0);
	bottom_data_size.resize(1);
	bottom_data_size[0] = pNetResource->feature_vector_size[index];

	top_data_size = bottom_data_size;
	alpha_d = inputparam.lrn_param().alpha();
	beta_d = inputparam.lrn_param().beta();
	localsize_d = inputparam.lrn_param().local_size();
	k_d = inputparam.lrn_param().k();
	normalize_type = inputparam.lrn_param().norm_region();

	return CUDA_RETURN_VALUE;
}
int HolidayLocalResponseNormalizeGPU::Exit()
{
	return CUDA_RETURN_VALUE;
}
int HolidayLocalResponseNormalizeGPU::GetTopSize(std::vector<HolidayDataSize>& out_data_size)
{
	out_data_size = top_data_size;
	return CUDA_RETURN_VALUE;
}
int HolidayLocalResponseNormalizeGPU::Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map)
{
#ifdef _DEBUG
	cudaEvent_t start1;
	int return_result= cudaEventCreate(&start1);
	cudaEvent_t stop1;
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, NULL);
#endif
	int dwsize = top_data_size[0].data_dim[1] * top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3];
	input_data_map[0]->m_gpu.Gpu_DataIn(pNetResourceGpu, input_data_map[0]->dwStorageType, input_data_map[0]->m_cpu.dataMemoryPtr());
	gLocalResponseNormalize_kernel << <CUDA_BLOCK(dwsize * input_data_map[0]->m_gpu.shape_[0], CUDA_THREAD_NUM), CUDA_THREAD_NUM, 0, pNetResourceGpu->main_stream>> >
		(input_data_map[0]->m_gpu.pfData_gpu, output_data_map[0]->m_gpu.pfData_gpu, dwsize * input_data_map[0]->m_gpu.shape_[0],
		top_data_size[0].data_dim[2], top_data_size[0].data_dim[3], top_data_size[0].data_dim[1], alpha_d / localsize_d, beta_d, k_d, localsize_d);
	output_data_map[0]->dwStorageType = DATA_GPU;

	output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];
	output_data_map[0]->data_shape[1] = top_data_size[0].data_dim[1];
	output_data_map[0]->data_shape[2] = top_data_size[0].data_dim[2];
	output_data_map[0]->data_shape[3] = top_data_size[0].data_dim[3];

#ifdef _DEBUG
	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
	float msecTotal1 = 0.0f;
	cudaEventElapsedTime(&msecTotal1, start1, stop1);
	printf(" LocalResponseNormalize: %f ms \n", msecTotal1);
#endif
#ifdef _DEBUG
	int dwSize = top_data_size[0].data_dim[1] * top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3];
	float *pfDataOut = new float[dwSize];
	cudaMemcpy(pfDataOut, output_data_map[0]->m_gpu.pfData_gpu,dwSize * sizeof(float), cudaMemcpyDeviceToHost);
	delete[] pfDataOut;
	cudaDeviceSynchronize();
	printf("LocalResponseNormalize:%s\n", cudaGetErrorString(cudaGetLastError()));
#endif
	return CUDA_RETURN_VALUE;
}
