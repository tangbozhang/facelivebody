#include "HolidayScaleGPU.h"


HolidayScaleGPU::HolidayScaleGPU()
{
	pfScale_d = 0;
	pfBias_d = 0;
}
HolidayScaleGPU::~HolidayScaleGPU()
{
}

__global__ static void gScale_kernel(float *pfDataIn, float *pfDataOut, int dwSize,
	int dwRow, int dwCol, int dwSlice, float *pfScale, float *pfBias)
{
	int dwIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (dwIdx < dwSize)
	{
		int dwDimS = dwIdx % (dwSlice * dwRow * dwCol) / (dwRow * dwCol);
		pfDataOut[dwIdx] = fmaf(pfDataIn[dwIdx], pfScale[dwDimS], pfBias[dwDimS]);
	}
}

int HolidayScaleGPU::Init(Holiday_LayerParameter &inputparam, HolidayNetResource<float> *pNetResource)
{
	pNetResourceGpu = (HolidayNetResourceGpu *)pNetResource->pNetResourceGpu;
	
	int index = inputparam.bottom_index(0);
	bottom_data_size.resize(1);
	bottom_data_size[0] = pNetResource->feature_vector_size[index];

	top_data_size = bottom_data_size;
	const float *pfscale_param = inputparam.mutable_scale_param()->mutable_scale_param()->mutable_data()->mutable_data();
	CUDA_ERROR(SafeCudaMalloc((void **)&pfScale_d, bottom_data_size[0].data_dim[1] * sizeof(float)));
	CUDA_ERROR(SafeCudaMalloc((void **)&pfBias_d, bottom_data_size[0].data_dim[1] * sizeof(float)));
	CUDA_ERROR(cudaMemcpyAsync(pfScale_d, pfscale_param, bottom_data_size[0].data_dim[1] * sizeof(float), cudaMemcpyHostToDevice, pNetResourceGpu->main_stream));
	
	const float *pfbias_param = inputparam.mutable_scale_param()->mutable_bias_param()->mutable_data()->mutable_data();

	if (inputparam.scale_param().bias_param().data().size())
	{
		CUDA_ERROR(cudaMemcpyAsync(pfBias_d, pfbias_param, bottom_data_size[0].data_dim[1] * sizeof(float), cudaMemcpyHostToDevice, pNetResourceGpu->main_stream));
	}
	else
	{
		CUDA_ERROR(cudaMemsetAsync(pfBias_d, 0, bottom_data_size[0].data_dim[1] * sizeof(float), pNetResourceGpu->main_stream));
	}

	cudaStreamSynchronize(pNetResourceGpu->main_stream);
	return CUDA_RETURN_VALUE;
}
int HolidayScaleGPU::Exit()
{
	if (pfScale_d) cudaFree(pfScale_d);
	if (pfBias_d) cudaFree(pfBias_d);

	return CUDA_RETURN_VALUE;
}

int HolidayScaleGPU::Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map)
{
#ifdef _DEBUG
	cudaEvent_t start1;
	cudaEventCreate(&start1);
	cudaEvent_t stop1;
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, NULL);
#endif
	int dwsize1 = top_data_size[0].data_dim[1] * top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3];

	input_data_map[0]->m_gpu.shape_ = input_data_map[0]->data_shape;
	input_data_map[0]->m_gpu.Gpu_DataIn(pNetResourceGpu, input_data_map[0]->dwStorageType, input_data_map[0]->m_cpu.dataMemoryPtr());
	input_data_map[0]->dwStorageType = DATA_GPU;
	output_data_map[0]->dwStorageType = DATA_GPU;
	output_data_map[0]->data_shape = input_data_map[0]->data_shape;
	output_data_map[0]->m_gpu.shape_ = output_data_map[0]->data_shape;
	output_data_map[0]->m_gpu.data_size = input_data_map[0]->m_gpu.data_size;
	gScale_kernel << <CUDA_BLOCK(output_data_map[0]->m_gpu.data_size, CUDA_THREAD_NUM), CUDA_THREAD_NUM, 0, pNetResourceGpu->main_stream>> >
		((float *)input_data_map[0]->m_gpu.pfData_gpu, (float *)output_data_map[0]->m_gpu.pfData_gpu,
		output_data_map[0]->m_gpu.data_size,
		output_data_map[0]->data_shape[2], output_data_map[0]->data_shape[3], output_data_map[0]->data_shape[1], pfScale_d, pfBias_d);

#ifdef _DEBUG
	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
	float msecTotal1 = 0.0f;
	cudaEventElapsedTime(&msecTotal1, start1, stop1);
	printf(" Scale: %f ms \n", msecTotal1);
#endif
#ifdef _DEBUG
	int dwsize = top_data_size[0].data_dim[1] * top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3];
	float *pfDataOut = new float[dwsize];
	cudaMemcpy(pfDataOut, output_data_map[0]->m_gpu.pfData_gpu,dwsize * sizeof(float), cudaMemcpyDeviceToHost);
	delete[] pfDataOut;
	cudaDeviceSynchronize();
	printf("Scale:%s\n", cudaGetErrorString(cudaGetLastError()));
#endif
	return CUDA_RETURN_VALUE;
}
