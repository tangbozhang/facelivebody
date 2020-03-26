#include "HolidayMemoryDataLayerGPU.h"
#include "device_launch_parameters.h"
#include "HolidayFeatureMap.h"

enum HolidayMeanType
{
	HolidayNoMean = 0,
	HolidayMeanFile = 1,
	HolidayMeanValue = 2,
};
HolidayMemoryDataLayerGPU::HolidayMemoryDataLayerGPU()
{
	pfMean_d = 0;
}
HolidayMemoryDataLayerGPU::~HolidayMemoryDataLayerGPU()
{
	
}
__global__ static void gMemoryDataNoMean_kernel(float *pfDataIn, float *pfDataOut, float dwSize, float fScale)
{
	int dwIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (dwIdx < dwSize)
	{
		pfDataOut[dwIdx] = pfDataIn[dwIdx] * fScale;
	}
}
__global__ static void gMemoryDataMean_kernel(float *pfDataIn, float *pfDataOut, int dwSize, int dwInnerDim, int dwMeanDim, float fScale, float *pfmean)
{
	int dwIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (dwIdx < dwSize)
	{
		pfDataOut[dwIdx] = (pfDataIn[dwIdx] - pfmean[(dwIdx / dwInnerDim) % dwMeanDim]) * fScale;
	}
}
int HolidayMemoryDataLayerGPU::Init(Holiday_LayerParameter&inputparam, HolidayNetResource<float> *pNetResource)
{
	pNetResourceGpu = (HolidayNetResourceGpu*)pNetResource->pNetResourceGpu;

	this->bottom_data_size.resize(1);
	this->bottom_data_size[0].data_dim.resize(4);
	this->bottom_data_size[0].data_dim[0] = pNetResource->max_batch_size;
	this->bottom_data_size[0].data_dim[1] = inputparam.memory_data_param().channels();

	int local_input_height = pNetResource->m_new_height > 0
		? pNetResource->m_new_height
		: int(inputparam.memory_data_param().height());
	int local_input_width = pNetResource->m_new_width > 0
		? pNetResource->m_new_width
		: int(inputparam.memory_data_param().width());

	this->bottom_data_size[0].data_dim[2] = local_input_height;
	this->bottom_data_size[0].data_dim[3] = local_input_width;


	this->top_data_size.resize(2);

	for (int i = 0; i < 2; i++)
	{
		this->top_data_size[i].data_dim.resize(4);
	}
	this->top_data_size[0].data_dim[0] = pNetResource->max_batch_size;
	this->top_data_size[0].data_dim[1] = this->bottom_data_size[0].data_dim[1];
	this->top_data_size[0].data_dim[2] = this->bottom_data_size[0].data_dim[2];
	this->top_data_size[0].data_dim[3] = this->bottom_data_size[0].data_dim[3];

	this->top_data_size[1].data_dim[0] = pNetResource->max_batch_size;
	this->top_data_size[1].data_dim[1] = 1;
	this->top_data_size[1].data_dim[2] = 1;
	this->top_data_size[1].data_dim[3] = 1;


	fScale = inputparam.memory_data_param().scale();
	//dwType 

	if (inputparam.memory_data_param().has_mean_file())
	{
		m_mean_type = 2;
	}
	else if (inputparam.memory_data_param().mean_value_size()>0)
	{
		m_mean_type = 1;
	}
	else
	{
		m_mean_type = 0;
	}
	if (HolidayMeanFile == m_mean_type)
	{
		int dwsize = top_data_size[0].data_dim[1] * top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3];
		CUDA_ERROR(SafeCudaMalloc((void **)&pfMean_d,dwsize * sizeof(float)));
		//float *pfMeanT;
		const float *pfMeanT = inputparam.mutable_memory_data_param()->mutable_mean_file()->mutable_data()->mutable_data();
		CUDA_ERROR(cudaMemcpyAsync(pfMean_d, pfMeanT, dwsize * sizeof(float), cudaMemcpyHostToDevice, pNetResourceGpu->main_stream));
		cudaStreamSynchronize(pNetResourceGpu->main_stream);
		delete[]pfMeanT;
	}
	else if (HolidayMeanValue == m_mean_type)
	{
		//float *pfMeanT;
		const float *pfMeanT = inputparam.mutable_memory_data_param()->mutable_mean_value()->mutable_data();
		CUDA_ERROR(SafeCudaMalloc((void **)&pfMean_d, bottom_data_size[0].data_dim[1] * sizeof(float)));
		CUDA_ERROR(cudaMemcpyAsync(pfMean_d, pfMeanT, bottom_data_size[0].data_dim[1] * sizeof(float), cudaMemcpyHostToDevice, pNetResourceGpu->main_stream));
		cudaStreamSynchronize(pNetResourceGpu->main_stream);
	}
	else
	{
		pfMean_d = 0;
	}
	return CUDA_RETURN_VALUE;
}
int HolidayMemoryDataLayerGPU::Exit()
{
	if (pfMean_d) cudaFree(pfMean_d);
	return CUDA_RETURN_VALUE;
}

int HolidayMemoryDataLayerGPU::Process(std::vector<HolidayFeatureMap<float>*>bottom_data_map, std::vector<HolidayFeatureMap<float>*>& top_data_map)
{
#ifdef _DEBUG
	cudaEvent_t start1;
	cudaEventCreate(&start1);
	cudaEvent_t stop1;
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, NULL);
#endif
	bottom_data_map[0]->m_gpu.Gpu_DataIn(pNetResourceGpu, bottom_data_map[0]->dwStorageType, bottom_data_map[0]->m_cpu.dataMemoryPtr());
	
	int dwSize = top_data_size[0].data_dim[1] * top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3];
	if (HolidayNoMean == m_mean_type)
	{///dwSize*
		
		gMemoryDataNoMean_kernel << <CUDA_BLOCK(dwSize* bottom_data_map[0]->m_gpu.shape_[0], CUDA_THREAD_NUM), CUDA_THREAD_NUM, 0, pNetResourceGpu->main_stream>> >
			(bottom_data_map[0]->m_gpu.pfData_gpu, top_data_map[0]->m_gpu.pfData_gpu, dwSize* bottom_data_map[0]->m_gpu.shape_[0], fScale);
	}
	else if (HolidayMeanFile == m_mean_type)
	{
		
		gMemoryDataMean_kernel << <CUDA_BLOCK(dwSize * bottom_data_map[0]->m_gpu.shape_[0], CUDA_THREAD_NUM), CUDA_THREAD_NUM, 0, pNetResourceGpu->main_stream>> >
			(bottom_data_map[0]->m_gpu.pfData_gpu, top_data_map[0]->m_gpu.pfData_gpu, dwSize * bottom_data_map[0]->m_gpu.shape_[0], 1, dwSize, fScale, pfMean_d);
	}
	else if (HolidayMeanValue == m_mean_type)
	{
		gMemoryDataMean_kernel << <CUDA_BLOCK(dwSize * bottom_data_map[0]->m_gpu.shape_[0], CUDA_THREAD_NUM), CUDA_THREAD_NUM, 0, pNetResourceGpu->main_stream>> >
			(bottom_data_map[0]->m_gpu.pfData_gpu, top_data_map[0]->m_gpu.pfData_gpu, dwSize * bottom_data_map[0]->m_gpu.shape_[0], top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3], top_data_size[0].data_dim[1], fScale, pfMean_d);
	}
	else
	{
	}
	top_data_map[0]->dwStorageType = DATA_GPU;
	top_data_map[0]->m_gpu.shape_[0] = bottom_data_map[0]->m_gpu.shape_[0];
	top_data_map[0]->data_shape[0] = top_data_map[0]->m_gpu.shape_[0];
	top_data_map[0]->data_shape[0] = bottom_data_map[0]->data_shape[0];
	top_data_map[0]->data_shape[1] = top_data_size[0].data_dim[1];
	top_data_map[0]->data_shape[2] = top_data_size[0].data_dim[2];
	top_data_map[0]->data_shape[3] = top_data_size[0].data_dim[3];

#ifdef _DEBUG
	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
	float msecTotal1 = 0.0f;
	cudaEventElapsedTime(&msecTotal1, start1, stop1);
	printf(" MemoryData: %f ms \n", msecTotal1);
#endif
#ifdef _DEBUG
	int dwsize = top_data_size[0].data_dim[1] * top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3];
	float *pfDataOut = new float[dwsize];
	cudaMemcpy(pfDataOut, top_data_map[0]->m_gpu.pfData_gpu,dwsize * sizeof(float), cudaMemcpyDeviceToHost);
	delete[] pfDataOut;
	cudaDeviceSynchronize();
	printf("MemoryData:%s\n", cudaGetErrorString(cudaGetLastError()));
#endif
	return CUDA_RETURN_VALUE;
}

