#include "HolidaySplitGPU.h"

HolidaySplitGPU::HolidaySplitGPU()
{
}
HolidaySplitGPU::~HolidaySplitGPU()
{
}
int HolidaySplitGPU::Init(Holiday_LayerParameter& inputparam, HolidayNetResource<float> *pNetResource)
{
	pNetResourceGpu = (HolidayNetResourceGpu *)pNetResource->pNetResourceGpu;
	//bottom_data_size = inputparam.bottom_data_size;
	int index = inputparam.bottom_index(0);
	bottom_data_size.resize(1);
	bottom_data_size[0] = pNetResource->feature_vector_size[index];

	top_data_size.resize(inputparam.top_index().size());
	for (int i = 0; i < inputparam.top_index().size(); i++)
	{
		top_data_size[i] = bottom_data_size[0];
	}
#ifdef _DEBUG
	cudaDeviceSynchronize();
	printf("Split Init:%s\n", cudaGetErrorString(cudaGetLastError()));
#endif
	return CUDA_RETURN_VALUE;
}
int HolidaySplitGPU::Exit()
{
	return CUDA_RETURN_VALUE;
}
//int Split_gpu::GetTopSize(std::vector<DataSize>& out_data_size)
//{
//	out_data_size = top_data_size;
//	return CUDA_RETURN_VALUE;
//}
int HolidaySplitGPU::Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map)
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
	int all_counts = 1;
	for (size_t i = 0; i < input_data_map[0]->data_shape.size(); i++)
	{
		all_counts *= input_data_map[0]->data_shape[i];
	}
	
	for (int i = 0; i < output_data_map.size(); i++)
	{
		cudaMemcpyAsync(output_data_map[i]->m_gpu.pfData_gpu, (float *)input_data_map[0]->m_gpu.pfData_gpu, all_counts* sizeof(float), cudaMemcpyDeviceToDevice, pNetResourceGpu->main_stream);
		output_data_map[i]->dwStorageType = DATA_GPU;
		output_data_map[i]->data_shape = input_data_map[0]->data_shape;
		output_data_map[i]->m_gpu.shape_ = output_data_map[i]->data_shape;
		output_data_map[i]->m_gpu.data_size = input_data_map[0]->m_gpu.data_size;
	}
#ifdef _DEBUG
	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
	float msecTotal1 = 0.0f;
	cudaEventElapsedTime(&msecTotal1, start1, stop1);
	printf(" Split: %f ms \n", msecTotal1);
#endif
#ifdef _DEBUG
	int dwSize2 = top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3] * top_data_size[0].data_dim[1];
	float *pfDataOut = new float[dwSize2];
	cudaMemcpy(pfDataOut, output_data_map[0]->m_gpu.pfData_gpu, dwSize2 * sizeof(float), cudaMemcpyDeviceToHost);
	delete[] pfDataOut;
	cudaDeviceSynchronize();
	printf("Split:%s\n", cudaGetErrorString(cudaGetLastError()));
#endif
	return CUDA_RETURN_VALUE;
}
