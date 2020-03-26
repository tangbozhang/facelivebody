#include "HolidayPreReluGPU.h"

HolidayPreReluGPU::HolidayPreReluGPU()
{
	pfSlope_d = 0;
}
HolidayPreReluGPU::~HolidayPreReluGPU()
{
}
__global__ static void gPreRelu_kernel(float *pfDataIn, float *pfDataOut, int dwSize, int dwRow, int dwCol, int dwSlice, float *pfSlope)
{
	int dwIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (dwIdx < dwSize)
	{
		int dwDimS = dwIdx % (dwSlice * dwRow * dwCol) / (dwRow * dwCol);
		//pfDataOut[dwIdx] = fmaf(fminf(pfDataIn[dwIdx], 0), pfSlope[dwDimS], fmaxf(pfDataIn[dwIdx], 0));
		pfDataOut[dwIdx] = pfDataIn[dwIdx] > 0 ? pfDataIn[dwIdx] : pfSlope[dwDimS] * pfDataIn[dwIdx];
	}
}

int HolidayPreReluGPU::Init(Holiday_LayerParameter& inputparam, HolidayNetResource<float> *pNetResource)
{
	pNetResourceGpu = (HolidayNetResourceGpu *)pNetResource->pNetResourceGpu;
	
	int index = inputparam.bottom_index(0);
	bottom_data_size.resize(1);
	bottom_data_size[0] = pNetResource->feature_vector_size[index];
	
	const float *pfprelu_param = inputparam.mutable_prelu_param()->mutable_param()->mutable_data()->mutable_data();
	top_data_size = bottom_data_size;
	CUDA_ERROR(SafeCudaMalloc((void **)&pfSlope_d, top_data_size[0].data_dim[1] * sizeof(float)));
	CUDA_ERROR(cudaMemcpyAsync(pfSlope_d, pfprelu_param, top_data_size[0].data_dim[1] * sizeof(float), cudaMemcpyHostToDevice, pNetResourceGpu->main_stream));//////////

	cudaStreamSynchronize(pNetResourceGpu->main_stream);
	return CUDA_RETURN_VALUE;
}
int HolidayPreReluGPU::Exit()
{
	if (pfSlope_d) cudaFree(pfSlope_d);

	return CUDA_RETURN_VALUE;
}

int HolidayPreReluGPU::Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map)
{
#ifdef _DEBUG
	cudaEvent_t start1;
	cudaEventCreate(&start1);
	cudaEvent_t stop1;
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, NULL);
#endif
	input_data_map[0]->m_gpu.shape_ = input_data_map[0]->data_shape;
	input_data_map[0]->m_gpu.Gpu_DataIn(pNetResourceGpu, input_data_map[0]->dwStorageType, input_data_map[0]->m_cpu.dataMemoryPtr());
	input_data_map[0]->dwStorageType = DATA_GPU;
	output_data_map[0]->dwStorageType = DATA_GPU;
	output_data_map[0]->data_shape = input_data_map[0]->data_shape;
	output_data_map[0]->m_gpu.shape_ = output_data_map[0]->data_shape;
	output_data_map[0]->m_gpu.data_size = input_data_map[0]->m_gpu.data_size;
	gPreRelu_kernel << <CUDA_BLOCK(output_data_map[0]->m_gpu.data_size, CUDA_THREAD_NUM), CUDA_THREAD_NUM, 0, pNetResourceGpu->main_stream>> >
		((float *)input_data_map[0]->m_gpu.pfData_gpu, (float *)output_data_map[0]->m_gpu.pfData_gpu, output_data_map[0]->m_gpu.data_size,
		output_data_map[0]->data_shape[2], output_data_map[0]->data_shape[3], output_data_map[0]->data_shape[1], pfSlope_d);

#ifdef _DEBUG
	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
	float msecTotal1 = 0.0f;
	cudaEventElapsedTime(&msecTotal1, start1, stop1);
	printf(" PreRelu: %f ms \n", msecTotal1);
#endif
#ifdef _DEBUG
	int dwSize2 = top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3] * top_data_size[0].data_dim[1];
	float *pfDataOut = new float[dwSize2];
	cudaMemcpy(pfDataOut, output_data_map[0]->m_gpu.pfData_gpu, dwSize2 * sizeof(float), cudaMemcpyDeviceToHost);
	delete[] pfDataOut;
	cudaDeviceSynchronize();
	printf("PreRelu:%s\n", cudaGetErrorString(cudaGetLastError()));
#endif
	return CUDA_RETURN_VALUE;
}
