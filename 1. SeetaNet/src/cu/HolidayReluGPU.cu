#include"HolidayReluGPU.h"
#include "HolidayCommonCuda.h"

HolidayReluGPU::HolidayReluGPU()
{
}
HolidayReluGPU::~HolidayReluGPU()
{
}
__global__ static void gRelu_kernel(float *pfDataIn, float *pfDataOut, int dwSize, float fNegative_slope)
{
	int dwIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (dwIdx < dwSize)
	{
		//pfDataOut[dwIdx] = fmaf(fminf(pfDataIn[dwIdx], 0), fNegative_slope, fmaxf(pfDataIn[dwIdx], 0));
		pfDataOut[dwIdx] = pfDataIn[dwIdx] > 0 ? pfDataIn[dwIdx] : fNegative_slope * pfDataIn[dwIdx];
	}
}
__global__ static void gReluCut_kernel(float *pfDataIn, float *pfDataOut, int dwSize, float fNegative_slope, float max)
{
	int dwIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (dwIdx < dwSize)
    {
        pfDataOut[dwIdx] = pfDataIn[dwIdx] > 0 ? pfDataIn[dwIdx] : fNegative_slope * pfDataIn[dwIdx];
        pfDataOut[dwIdx] = pfDataOut[dwIdx] > max ? max : pfDataOut[dwIdx];
	}
}
int HolidayReluGPU::Init(Holiday_LayerParameter& inputparam, HolidayNetResource<float> *pNetResource)
{
	pNetResourceGpu = (HolidayNetResourceGpu *)pNetResource->pNetResourceGpu;
	
	int index = inputparam.bottom_index(0);
	bottom_data_size.resize(1);
	bottom_data_size[0] = pNetResource->feature_vector_size[index];

	top_data_size = bottom_data_size;
	fNegative_slope = inputparam.relu_param().negative_slope();

    m_has_max = inputparam.relu_param().has_max();
    if (m_has_max)
    {
        m_max = inputparam.relu_param().max();
    }

	return CUDA_RETURN_VALUE;
}
int HolidayReluGPU::Exit()
{
	return CUDA_RETURN_VALUE;
}

int HolidayReluGPU::Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map)
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

    if (m_has_max)
    {
        gReluCut_kernel << <CUDA_BLOCK(output_data_map[0]->m_gpu.data_size, CUDA_THREAD_NUM), CUDA_THREAD_NUM, 0, pNetResourceGpu->main_stream>> >
            ((float *)input_data_map[0]->m_gpu.pfData_gpu, (float *)output_data_map[0]->m_gpu.pfData_gpu, output_data_map[0]->m_gpu.data_size, fNegative_slope, m_max);
    }
    else
    {
        gRelu_kernel << <CUDA_BLOCK(output_data_map[0]->m_gpu.data_size, CUDA_THREAD_NUM), CUDA_THREAD_NUM, 0, pNetResourceGpu->main_stream>> >
            ((float *)input_data_map[0]->m_gpu.pfData_gpu, (float *)output_data_map[0]->m_gpu.pfData_gpu, output_data_map[0]->m_gpu.data_size, fNegative_slope);
    }



#ifdef _DEBUG
	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
	float msecTotal1 = 0.0f;
	cudaEventElapsedTime(&msecTotal1, start1, stop1);
	printf(" Relu: %f ms \n", msecTotal1);
#endif
#ifdef _DEBUG
	int dwSize1 = top_data_size[0].data_dim[1] * top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3];
	float *pfDataOut = new float[dwSize1];
	cudaMemcpy(pfDataOut, output_data_map[0]->m_gpu.pfData_gpu, dwSize1 * sizeof(float), cudaMemcpyDeviceToHost);
	delete[] pfDataOut;
	cudaDeviceSynchronize();
	printf("Relu:%s\n", cudaGetErrorString(cudaGetLastError()));
#endif
	return CUDA_RETURN_VALUE;
}
