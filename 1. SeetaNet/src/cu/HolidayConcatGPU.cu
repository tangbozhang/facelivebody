#include"HolidayConcatGPU.h"


HolidayConcatGPU::HolidayConcatGPU()
{
	pdwWidthIn = 0;
}
HolidayConcatGPU::~HolidayConcatGPU()
{
}
__global__ static void gConcat_kernel(float *pfDataIn, float *pfDataOut, int dwWidthIn, int dwWidthOut, int dwSize)
{
	int dwIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int dwIdxX = dwIdx % dwWidthIn;
	int dwIdxY = dwIdx / dwWidthIn;
	if (dwIdx < dwSize)
	{
		pfDataOut[dwIdxY * dwWidthOut + dwIdxX] = pfDataIn[dwIdx];
	}
}

int HolidayConcatGPU::Init(Holiday_LayerParameter& inputparam, HolidayNetResource<float> *pNetResource)
{
	pNetResourceGpu = (HolidayNetResourceGpu *)pNetResource->pNetResourceGpu;

    int bottom_length = inputparam.bottom_index().size();
    this->bottom_data_size.resize(bottom_length);
    for (size_t i = 0; i < bottom_length; i++)
    {
        int index = inputparam.bottom_index(i);
        this->bottom_data_size[i] = pNetResource->feature_vector_size[index];
    }

	top_data_size.resize(1);
    top_data_size[0].data_dim.resize(4);
	top_data_size[0].data_dim[1] = 0;
	top_data_size[0].data_dim[2] = 0;
	top_data_size[0].data_dim[3] = 0;
	pdwWidthIn = (int *)malloc(bottom_data_size.size() * sizeof(int));
	dwConcatAxis = inputparam.concat_param().axis();
	switch (inputparam.concat_param().axis())
	{
	case 1:
		
		for (int i = 0; i < bottom_data_size.size(); i++)
		{
			int dwSize1 = bottom_data_size[i].data_dim[2] * bottom_data_size[i].data_dim[3] * bottom_data_size[i].data_dim[1];
			top_data_size[0].data_dim[1] += bottom_data_size[i].data_dim[1];
			pdwWidthIn[i] = dwSize1;
		}
		top_data_size[0].data_dim[2] = bottom_data_size[0].data_dim[2];
		top_data_size[0].data_dim[3] = bottom_data_size[0].data_dim[3];
		dwWidthOut = top_data_size[0].data_dim[1] * top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3];
		break;
	case 2:
		for (int i = 0; i < bottom_data_size.size(); i++)
		{
			top_data_size[0].data_dim[2] += bottom_data_size[i].data_dim[2];
			pdwWidthIn[i] = bottom_data_size[i].data_dim[2] * bottom_data_size[i].data_dim[3];
		}
		top_data_size[0].data_dim[1] = bottom_data_size[0].data_dim[1];
		top_data_size[0].data_dim[3] = bottom_data_size[0].data_dim[3];
		dwWidthOut = top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3];
		break;
	case 3:
		for (int i = 0; i < bottom_data_size.size(); i++)
		{
			top_data_size[0].data_dim[3] += bottom_data_size[i].data_dim[3];
			pdwWidthIn[i] = bottom_data_size[i].data_dim[3];
		}
		top_data_size[0].data_dim[1] = bottom_data_size[0].data_dim[1];
		top_data_size[0].data_dim[2] = bottom_data_size[0].data_dim[2];
		dwWidthOut = top_data_size[0].data_dim[3];
		break;
	default:
		break;
	}
    top_data_size[0].data_dim[0] = this->bottom_data_size[0].data_dim[0];
	int dwSize = top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3] * top_data_size[0].data_dim[1];
#ifdef _DEBUG
	cudaDeviceSynchronize();
	printf("Concat Init:%s\n", cudaGetErrorString(cudaGetLastError()));
#endif
	return CUDA_RETURN_VALUE;
}
int HolidayConcatGPU::Exit()
{
	if (pdwWidthIn) free(pdwWidthIn);
	return CUDA_RETURN_VALUE;
}

int HolidayConcatGPU::Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map)
{
#ifdef _DEBUG
	cudaEvent_t start1;
	cudaEventCreate(&start1);
	cudaEvent_t stop1;
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, NULL);

#endif	
	for (int i = 0; i < input_data_map.size(); i++)
	{
		input_data_map[i]->m_gpu.shape_ = input_data_map[i]->data_shape;
		input_data_map[i]->m_gpu.Gpu_DataIn(pNetResourceGpu, input_data_map[i]->dwStorageType, input_data_map[i]->m_cpu.dataMemoryPtr());
		input_data_map[i]->dwStorageType = DATA_GPU;
		int j = 3;
		pdwWidthIn[i] = 1;
		while (j >= dwConcatAxis)
		{
			pdwWidthIn[i] *= input_data_map[i]->data_shape[j];
			j--;
		}
	}
	output_data_map[0]->dwStorageType = DATA_GPU;
	output_data_map[0]->data_shape = input_data_map[0]->data_shape;
	for (int i = 1; i < input_data_map.size(); ++i)
	{
		output_data_map[0]->data_shape[dwConcatAxis] += input_data_map[i]->data_shape[dwConcatAxis];
	}
	output_data_map[0]->m_gpu.shape_ = output_data_map[0]->data_shape;
	output_data_map[0]->m_gpu.data_size = output_data_map[0]->data_shape[0] * output_data_map[0]->data_shape[1] * output_data_map[0]->data_shape[2] * output_data_map[0]->data_shape[3];
	float *pfCurDataOut = (float *)output_data_map[0]->m_gpu.pfData_gpu;
	for (int i = 0; i < input_data_map.size(); i++)
	{
		gConcat_kernel << <CUDA_BLOCK(input_data_map[i]->m_gpu.data_size, CUDA_THREAD_NUM), CUDA_THREAD_NUM, 0, pNetResourceGpu->main_stream>> >
			((float *)input_data_map[i]->m_gpu.pfData_gpu, pfCurDataOut, pdwWidthIn[i], dwWidthOut, input_data_map[i]->m_gpu.data_size);
		pfCurDataOut += pdwWidthIn[i];
	}
#ifdef _DEBUG
	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
	float msecTotal1 = 0.0f;
	cudaEventElapsedTime(&msecTotal1, start1, stop1);
	printf(" Concat: %f ms\n ", msecTotal1);
#endif
#ifdef _DEBUG
	int dwSize3 = top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3] * top_data_size[0].data_dim[1];
	float *pfDataOut = new float[dwSize3];
	cudaMemcpy(pfDataOut, output_data_map[0]->m_gpu.pfData_gpu, dwSize3 * sizeof(float), cudaMemcpyDeviceToHost);
	delete[] pfDataOut;
	cudaDeviceSynchronize();
	printf("Concat:%s\n", cudaGetErrorString(cudaGetLastError()));
#endif
	return CUDA_RETURN_VALUE;
}
