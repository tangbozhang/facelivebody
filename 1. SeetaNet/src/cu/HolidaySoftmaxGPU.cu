#include "HolidaySoftmaxGPU.h"


#define SOFTMAX_BLOCK_SIZE 512
#define SOFTMAX_DIV_B_W 16

HolidaySoftMaxGPU::HolidaySoftMaxGPU()
{
	pfMax_d = 0;
	pfSum_d = 0;
}
HolidaySoftMaxGPU::~HolidaySoftMaxGPU()
{
}
__global__ static void gMax_kernel(float *pfDataIn, int dwSize, float *pfMax)
{
	__shared__ float temp_array[SOFTMAX_DIV_B_W];
	int wid = threadIdx.x >> 5;
	int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x + blockDim.x;
	float fA;
	if (idx < dwSize)
		fA = fmaxf(pfDataIn[idx - blockDim.x], pfDataIn[idx]);
	else if (idx - blockDim.x < dwSize)
		fA = pfDataIn[idx - blockDim.x];
	else
		fA = -3.402823E38;
	fA = fmaxf(fA, __shfl_down(fA, 16));
	fA = fmaxf(fA, __shfl_down(fA, 8));
	fA = fmaxf(fA, __shfl_down(fA, 4));
	fA = fmaxf(fA, __shfl_down(fA, 2));
	fA = fmaxf(fA, __shfl_down(fA, 1));
	if ((threadIdx.x & 31) == 0)
		temp_array[wid] = fA;
	__syncthreads();
	if (threadIdx.x < 8)
		temp_array[threadIdx.x] = fmaxf(temp_array[threadIdx.x], temp_array[threadIdx.x + 8]);
	if (threadIdx.x < 4)
		temp_array[threadIdx.x] = fmaxf(temp_array[threadIdx.x], temp_array[threadIdx.x + 4]);
	if (threadIdx.x < 2)
		temp_array[threadIdx.x] = fmaxf(temp_array[threadIdx.x], temp_array[threadIdx.x + 2]);
	if (threadIdx.x < 1)
		temp_array[threadIdx.x] = fmaxf(temp_array[threadIdx.x], temp_array[threadIdx.x + 1]);
	if (threadIdx.x == 0)
		//atomicMax(pfMax, temp_array[threadIdx.x]);
		atomicExch(pfMax, fmaxf(*pfMax, temp_array[threadIdx.x]));
	return;
}
__global__ static void gAdd_kernel(float *pfDataIn, int dwSize, float *pfMax, float *pfSum)
{
	__shared__ float temp_array[SOFTMAX_DIV_B_W];
	int wid = threadIdx.x >> 5;
	int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x + blockDim.x;
	float fA;
	if (idx < dwSize)
		fA = expf(pfDataIn[idx - blockDim.x] - pfMax[0]) + expf(pfDataIn[idx] - pfMax[0]);
	else if (idx - blockDim.x < dwSize)
		fA = expf(pfDataIn[idx - blockDim.x] - pfMax[0]);
	else
		fA = 0.f;
	fA += __shfl_down(fA, 16);
	fA += __shfl_down(fA, 8);
	fA += __shfl_down(fA, 4);
	fA += __shfl_down(fA, 2);
	fA += __shfl_down(fA, 1);
	if ((threadIdx.x & 31) == 0)
		temp_array[wid] = fA;
	__syncthreads();
	if (threadIdx.x < 8)
		temp_array[threadIdx.x] += temp_array[threadIdx.x + 8];
	if (threadIdx.x < 4)
		temp_array[threadIdx.x] += temp_array[threadIdx.x + 4];
	if (threadIdx.x < 2)
		temp_array[threadIdx.x] += temp_array[threadIdx.x + 2];
	if (threadIdx.x < 1)
		temp_array[threadIdx.x] += temp_array[threadIdx.x + 1];
	if (threadIdx.x == 0)
		atomicAdd(pfSum, temp_array[threadIdx.x]);
	return;
}
__global__ static void gDiv_kernel(float *pfDataIn, int dwSize, float *pfMax, float *pfSum, float *pfDataOut)
{
	int dwIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (dwIdx < dwSize)
	{
		pfDataOut[dwIdx] = expf(pfDataIn[dwIdx] - pfMax[0]) / pfSum[0];
	}
}


int HolidaySoftMaxGPU::Init(Holiday_LayerParameter& inputparam, HolidayNetResource<float> *pNetResource)
{
	pNetResourceGpu = (HolidayNetResourceGpu *)pNetResource->pNetResourceGpu;
	
	int index = inputparam.bottom_index(0);
	bottom_data_size.resize(1);
	bottom_data_size[0] = pNetResource->feature_vector_size[index];

	top_data_size = bottom_data_size;
	CUDA_ERROR(SafeCudaMalloc((void **)&pfMax_d, pNetResourceGpu->dwMaxBatchNum * sizeof(float)));
	CUDA_ERROR(SafeCudaMalloc((void **)&pfSum_d, pNetResourceGpu->dwMaxBatchNum * sizeof(float)));

	return CUDA_RETURN_VALUE;
}
int HolidaySoftMaxGPU::Exit()
{
	if (pfMax_d) cudaFree(pfMax_d);
	if (pfSum_d) cudaFree(pfSum_d);

	return CUDA_RETURN_VALUE;
}

int HolidaySoftMaxGPU::Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map)
{
#ifdef _DEBUG
	cudaEvent_t start1;
	cudaEventCreate(&start1);
	cudaEvent_t stop1;
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, NULL);
#endif
	input_data_map[0]->m_gpu.Gpu_DataIn(pNetResourceGpu, input_data_map[0]->dwStorageType, input_data_map[0]->m_cpu.dataMemoryPtr());
	input_data_map[0]->m_gpu.shape_ = input_data_map[0]->data_shape;
	input_data_map[0]->m_gpu.Gpu_DataIn(pNetResourceGpu, input_data_map[0]->dwStorageType, input_data_map[0]->m_cpu.dataMemoryPtr());
	input_data_map[0]->dwStorageType = DATA_GPU;
	output_data_map[0]->dwStorageType = DATA_GPU;
	output_data_map[0]->data_shape = input_data_map[0]->data_shape;
	output_data_map[0]->m_gpu.shape_ = output_data_map[0]->data_shape;
	output_data_map[0]->m_gpu.data_size = input_data_map[0]->m_gpu.data_size;
	int dwOffset = input_data_map[0]->data_shape[3] * input_data_map[0]->data_shape[2] * input_data_map[0]->data_shape[1];

	for (int i = 0; i < input_data_map[0]->data_shape[0]; i++)
	{
		// int dwIdx = i % pNetResourceGpu->dwStreamNum;
		float *pfCurDataIn_d = (float *)input_data_map[0]->m_gpu.pfData_gpu + i * dwOffset;
		float *pfCurDataOut_d = (float *)output_data_map[0]->m_gpu.pfData_gpu + i * dwOffset;
		//process
		CUDA_ERROR(cudaMemcpyAsync(pfMax_d + i, pfCurDataIn_d, sizeof(float), cudaMemcpyDeviceToDevice, pNetResourceGpu->main_stream));
		CUDA_ERROR(cudaMemsetAsync(pfSum_d + i, 0, sizeof(float), pNetResourceGpu->main_stream));
		gMax_kernel << <CUDA_BLOCK(dwOffset, SOFTMAX_BLOCK_SIZE), SOFTMAX_BLOCK_SIZE, 0, pNetResourceGpu->main_stream >> >(pfCurDataIn_d, dwOffset, pfMax_d + i);
		gAdd_kernel << <CUDA_BLOCK(dwOffset, SOFTMAX_BLOCK_SIZE), SOFTMAX_BLOCK_SIZE, 0, pNetResourceGpu->main_stream >> >(pfCurDataIn_d, dwOffset, pfMax_d + i, pfSum_d + i);
		gDiv_kernel << <CUDA_BLOCK(dwOffset, SOFTMAX_BLOCK_SIZE), SOFTMAX_BLOCK_SIZE, 0, pNetResourceGpu->main_stream >> >(pfCurDataIn_d, dwOffset, pfMax_d + i, pfSum_d + i, pfCurDataOut_d);
	}

#ifdef _DEBUG
	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
	float msecTotal1 = 0.0f;
	cudaEventElapsedTime(&msecTotal1, start1, stop1);
	printf(" SoftMax: %f ms \n", msecTotal1);
#endif
#ifdef _DEBUG
	int dwSize1 = top_data_size[0].data_dim[1] * top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3];
	float *pfDataOut = new float[dwSize1 * input_data_map[0]->m_gpu.shape_[0]];
	cudaMemcpy(pfDataOut, output_data_map[0]->m_gpu.pfData_gpu, dwSize1 * input_data_map[0]->m_gpu.shape_[0] * sizeof(float), cudaMemcpyDeviceToHost);
	delete[] pfDataOut;
	cudaDeviceSynchronize();
	printf("SoftMax:%s\n", cudaGetErrorString(cudaGetLastError()));
#endif
	return CUDA_RETURN_VALUE;
}
