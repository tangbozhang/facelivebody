#include"HolidayPowerGPU.h"

HolidayPowerGPU::HolidayPowerGPU()
{
}
HolidayPowerGPU::~HolidayPowerGPU()
{
}
__global__ static void gPower_kernel(float *pfDataIn, float *pfDataOut, int dwSize, float fScale, float fShift, float fPower)
{
	int dwIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (dwIdx < dwSize)
	{
		pfDataOut[dwIdx] = powf(fScale * pfDataIn[dwIdx] + fShift, fPower);
	}
}

int HolidayPowerGPU::Init(Holiday_LayerParameter& inputparam, HolidayNetResource<float> *pNetResource)
{
	pNetResourceGpu = (HolidayNetResourceGpu *)pNetResource->pNetResourceGpu;
	
	//bottom_data_size = inputparam.bottom_data_size;
	int index = inputparam.bottom_index(0);
	bottom_data_size.resize(1);
	bottom_data_size[0] = pNetResource->feature_vector_size[index];

	top_data_size = bottom_data_size;
	fScale = inputparam.power_param().scale();
	fShift = inputparam.power_param().shift();
	fPower = inputparam.power_param().power();
	return CUDA_RETURN_VALUE;
}
int HolidayPowerGPU::Exit()
{
	return CUDA_RETURN_VALUE;
}
//int Power_gpu::GetTopSize(std::vector<DataSize>& out_data_size)
//{
//	out_data_size = top_data_size;
//	return CUDA_RETURN_VALUE;
//}
int HolidayPowerGPU::Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map)
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
	gPower_kernel << <CUDA_BLOCK(output_data_map[0]->m_gpu.data_size, CUDA_THREAD_NUM), CUDA_THREAD_NUM, 0, pNetResourceGpu->main_stream>> >
		((float *)input_data_map[0]->m_gpu.pfData_gpu, (float *)output_data_map[0]->m_gpu.pfData_gpu, output_data_map[0]->m_gpu.data_size, fScale, fShift, fPower);

#ifdef _DEBUG
	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
	float msecTotal1 = 0.0f;
	cudaEventElapsedTime(&msecTotal1, start1, stop1);
	printf(" Power: %f ms \n", msecTotal1);
#endif
#ifdef _DEBUG
	int dwSize = top_data_size[0].data_dim[1] * top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3];
	float *pfDataOut = new float[dwSize];
	cudaMemcpy(pfDataOut, output_data_map[0]->m_gpu.pfData_gpu,dwSize * sizeof(float), cudaMemcpyDeviceToHost);
	delete[] pfDataOut;
	cudaDeviceSynchronize();
	printf("Power:%s\n", cudaGetErrorString(cudaGetLastError()));
#endif
	return CUDA_RETURN_VALUE;
}
