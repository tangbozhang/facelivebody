#include "HolidayEltwiseGPU.h"

enum EltwiseType
{
	EltwiseParameter_EltwiseOp_PROD = 0,
	EltwiseParameter_EltwiseOp_SUM = 1,
	EltwiseParameter_EltwiseOp_MAX = 2
};

HolidayEltwiseGPU::HolidayEltwiseGPU()
{
	ppfCurDataIn = 0;
	ppfCurDataIn_d = 0;
	pfCoeff_d = 0;
}
HolidayEltwiseGPU::~HolidayEltwiseGPU()
{
}
__global__ static void gEltwiseProd_kernel(float **ppfDataIn, float *pfDataOut, int dwSize, int dwNum)
{
	int dwIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (dwIdx < dwSize)
	{
		float fTmp = 1.f;
		for (int i = 0; i < dwNum; i++)
		{
			fTmp *= ppfDataIn[i][dwIdx];
		}
		pfDataOut[dwIdx] = fTmp;
	}
}
__global__ static void gEltwiseSum_kernel(float **ppfDataIn, float *pfDataOut, int dwSize, int dwNum, float *pfCoeff)
{
	int dwIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (dwIdx < dwSize)
	{
		float fTmp = 0.f;
		for (int i = 0; i < dwNum; i++)
		{
			fTmp += pfCoeff[i] * ppfDataIn[i][dwIdx];
		}
		pfDataOut[dwIdx] = fTmp;
	}
}
__global__ static void gEltwiseMax_kernel(float **ppfDataIn, float *pfDataOut, int dwSize, int dwNum)
{
	int dwIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (dwIdx < dwSize)
	{
		float fTmp = -3.402823E38;
		for (int i = 0; i < dwNum; i++)
		{
			fTmp = fmaxf(ppfDataIn[i][dwIdx], fTmp);
		}
		pfDataOut[dwIdx] = fTmp;
	}
}
int HolidayEltwiseGPU::Init(Holiday_LayerParameter& inputparam, HolidayNetResource<float> *pNetResource)
{
	dwType = inputparam.eltwise_param().operation();

	pNetResourceGpu = (HolidayNetResourceGpu *)pNetResource->pNetResourceGpu;
	int bottom_length = inputparam.bottom_index().size();
	this->bottom_data_size.resize(bottom_length);
	for (size_t i = 0; i < bottom_length; i++)
	{
		int index = inputparam.bottom_index(i);
		bottom_data_size[i] = pNetResource->feature_vector_size[index];
	}
	

	//top_data_size = bottom_data_size;
	top_data_size.clear();
	top_data_size.push_back(bottom_data_size[0]);
	eltwise_coeff.clear();
	if (inputparam.eltwise_param().coeff().empty())
	{
		int length_resize = bottom_length;

		eltwise_coeff.resize(length_resize, 1);
	}
	else
	{
		//eltwise_coeff = inputparam.eltwise_coeff;
		for (int i = 0; i < eltwise_coeff.size(); i++)
		{
			float m_value = 0.0;
			m_value = inputparam.eltwise_param().coeff(i);
			eltwise_coeff.push_back(m_value);
		}
		//eltwise_coeff = 
	}
	ppfCurDataIn = new float*[eltwise_coeff.size()];
	CUDA_ERROR(SafeCudaMalloc((void **)&ppfCurDataIn_d, eltwise_coeff.size() * sizeof(float*)));
	if (EltwiseParameter_EltwiseOp_SUM == dwType)
	{
		CUDA_ERROR(SafeCudaMalloc((void **)&pfCoeff_d, eltwise_coeff.size() * sizeof(float)));
		CUDA_ERROR(cudaMemcpyAsync(pfCoeff_d, &(eltwise_coeff[0]), eltwise_coeff.size() * sizeof(float), cudaMemcpyHostToDevice, pNetResourceGpu->main_stream));
	}
	else
	{
		pfCoeff_d = 0;
	}

	cudaStreamSynchronize(pNetResourceGpu->main_stream);
	return CUDA_RETURN_VALUE;

}
int HolidayEltwiseGPU::Exit()
{
	if (ppfCurDataIn) delete[]ppfCurDataIn;
	if (ppfCurDataIn_d) cudaFree(ppfCurDataIn_d);
	if (pfCoeff_d) cudaFree(pfCoeff_d);

	return CUDA_RETURN_VALUE;

}

int HolidayEltwiseGPU::Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map)
{
#ifdef _DEBUG
	cudaEvent_t start1;
	cudaEventCreate(&start1);
	cudaEvent_t stop1;
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, NULL);
#endif
	//data update
	for (int j = 0; j < input_data_map.size(); j++)
	{
		input_data_map[j]->m_gpu.shape_ = input_data_map[j]->data_shape;
		input_data_map[j]->m_gpu.Gpu_DataIn(pNetResourceGpu, input_data_map[j]->dwStorageType, input_data_map[j]->m_cpu.dataMemoryPtr());
		input_data_map[j]->dwStorageType = DATA_GPU;
		ppfCurDataIn[j] = (float *)input_data_map[j]->m_gpu.pfData_gpu;
	}
	CUDA_ERROR(cudaMemcpyAsync(ppfCurDataIn_d, ppfCurDataIn, input_data_map.size() * sizeof(float *), cudaMemcpyHostToDevice, pNetResourceGpu->main_stream));
	output_data_map[0]->dwStorageType = DATA_GPU;
	output_data_map[0]->data_shape = input_data_map[0]->data_shape;
	output_data_map[0]->m_gpu.shape_ = output_data_map[0]->data_shape;
	output_data_map[0]->m_gpu.data_size = input_data_map[0]->m_gpu.data_size;
	if (EltwiseParameter_EltwiseOp_PROD == dwType)
	{
		gEltwiseProd_kernel << <CUDA_BLOCK(output_data_map[0]->m_gpu.data_size, CUDA_THREAD_NUM), CUDA_THREAD_NUM, 0, pNetResourceGpu->main_stream>> >
			(ppfCurDataIn_d, (float *)output_data_map[0]->m_gpu.pfData_gpu, output_data_map[0]->m_gpu.data_size, input_data_map.size());
	}
	else if (EltwiseParameter_EltwiseOp_SUM == dwType)
	{
		gEltwiseSum_kernel << <CUDA_BLOCK(output_data_map[0]->m_gpu.data_size, CUDA_THREAD_NUM), CUDA_THREAD_NUM, 0, pNetResourceGpu->main_stream>> >
			(ppfCurDataIn_d, (float *)output_data_map[0]->m_gpu.pfData_gpu, output_data_map[0]->m_gpu.data_size, input_data_map.size(), pfCoeff_d);
	}
	else if (EltwiseParameter_EltwiseOp_MAX == dwType)
	{
		gEltwiseMax_kernel << <CUDA_BLOCK(output_data_map[0]->m_gpu.data_size, CUDA_THREAD_NUM), CUDA_THREAD_NUM, 0, pNetResourceGpu->main_stream>> >
			(ppfCurDataIn_d, (float *)output_data_map[0]->m_gpu.pfData_gpu, output_data_map[0]->m_gpu.data_size, input_data_map.size());
	}
	else
	{
	}

	output_data_map[0]->dwStorageType = DATA_GPU;
	output_data_map[0]->m_gpu.shape_[0] = input_data_map[0]->m_gpu.shape_[0];
	//output_data_map[0]->shape_[0] = output_data_map[0]->m_gpu.shape_[0];
	output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];
	output_data_map[0]->data_shape[1] = input_data_map[0]->data_shape[1];
	output_data_map[0]->data_shape[2] = input_data_map[0]->data_shape[2];
	output_data_map[0]->data_shape[3] = input_data_map[0]->data_shape[3];

#ifdef _DEBUG
	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
	float msecTotal1 = 0.0f;
	cudaEventElapsedTime(&msecTotal1, start1, stop1);
	printf(" Eltwise: %f ms \n", msecTotal1);
#endif
#ifdef _DEBUG

	int dwSize = top_data_size[0].data_dim[1] * top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3];
	float *pfDataOut = new float[dwSize];
	cudaMemcpy(pfDataOut, output_data_map[0]->m_gpu.pfData_gpu, dwSize * sizeof(float), cudaMemcpyDeviceToHost);
	delete[] pfDataOut;
	cudaDeviceSynchronize();
	printf("Eltwise:%s\n", cudaGetErrorString(cudaGetLastError()));
#endif
	return CUDA_RETURN_VALUE;
}
