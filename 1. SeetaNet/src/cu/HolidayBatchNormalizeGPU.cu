#include"HolidayBatchNormalizeGPU.h"
#include"HolidayCommonCuda.h"

HolidayBatchNormalizeGPU::HolidayBatchNormalizeGPU()
{
	pfMean_d = 0;
	pfVariance_d = 0;
}
HolidayBatchNormalizeGPU::~HolidayBatchNormalizeGPU()
{
}
__global__ static void gBatchNormalize_kernel(float *pfDataIn, float *pfDataOut, int dwSize,
	int dwRow, int dwCol, int dwSlice, float *pfMean, float *pfVariance)
{
	int dwIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (dwIdx < dwSize)
	{
		int dwDimS = dwIdx % (dwSlice * dwRow * dwCol) / (dwRow * dwCol);
		pfDataOut[dwIdx] = (pfDataIn[dwIdx] - pfMean[dwDimS]) / pfVariance[dwDimS];
	}
}

int HolidayBatchNormalizeGPU::Init(Holiday_LayerParameter& inputparam, HolidayNetResource<float> *pNetResource)
{
	pNetResourceGpu = (HolidayNetResourceGpu *)pNetResource->pNetResourceGpu;
	
	int index = inputparam.bottom_index(0);
	bottom_data_size.resize(1);
	bottom_data_size[0] = pNetResource->feature_vector_size[index];


	const float *pfmean_param = inputparam.mutable_batchnormlise_param()->mutable_mean_param()->mutable_data()->mutable_data();
	const float *pfcovariance_param = inputparam.mutable_batchnormlise_param()->mutable_covariance_param()->mutable_data()->mutable_data();


	top_data_size = bottom_data_size;
	CUDA_ERROR(SafeCudaMalloc((void **)&pfMean_d, bottom_data_size[0].data_dim[1] * sizeof(float)));
	CUDA_ERROR(SafeCudaMalloc((void **)&pfVariance_d, bottom_data_size[0].data_dim[1] * sizeof(float)));
	CUDA_ERROR(cudaMemcpyAsync(pfMean_d, pfmean_param, bottom_data_size[0].data_dim[1] * sizeof(float), cudaMemcpyHostToDevice, pNetResourceGpu->main_stream));
	CUDA_ERROR(cudaMemcpyAsync(pfVariance_d, pfcovariance_param, bottom_data_size[0].data_dim[1] * sizeof(float), cudaMemcpyHostToDevice, pNetResourceGpu->main_stream));

    cudaStreamSynchronize(pNetResourceGpu->main_stream);
	return CUDA_RETURN_VALUE;
}
int HolidayBatchNormalizeGPU::Exit()
{
	if (pfMean_d) cudaFree(pfMean_d);
	if (pfVariance_d) cudaFree(pfVariance_d);

	return CUDA_RETURN_VALUE;
}

int HolidayBatchNormalizeGPU::Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>&output_data_map)
{
#ifdef _DEBUG
	cudaEvent_t start1;
	cudaEventCreate(&start1);
	cudaEvent_t stop1;
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, NULL);
#endif
	int dwSize1 = top_data_size[0].data_dim[1] * top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3];
	input_data_map[0]->m_gpu.Gpu_DataIn(pNetResourceGpu, input_data_map[0]->dwStorageType, input_data_map[0]->m_cpu.dataMemoryPtr());
	gBatchNormalize_kernel << <CUDA_BLOCK(dwSize1 * input_data_map[0]->m_gpu.shape_[0], CUDA_THREAD_NUM), CUDA_THREAD_NUM, 0, pNetResourceGpu->main_stream>> >
		(input_data_map[0]->m_gpu.pfData_gpu, output_data_map[0]->m_gpu.pfData_gpu, dwSize1 * input_data_map[0]->m_gpu.shape_[0],
		top_data_size[0].data_dim[2], top_data_size[0].data_dim[3], top_data_size[0].data_dim[1], pfMean_d, pfVariance_d);
	output_data_map[0]->dwStorageType = DATA_GPU;

	output_data_map[0]->m_gpu.shape_[0] = input_data_map[0]->m_gpu.shape_[0];

	output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];
	output_data_map[0]->data_shape[1] = top_data_size[0].data_dim[1];
	output_data_map[0]->data_shape[2] = top_data_size[0].data_dim[2];
	output_data_map[0]->data_shape[3] = top_data_size[0].data_dim[3];

#ifdef _DEBUG
	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
	float msecTotal1 = 0.0f;
	cudaEventElapsedTime(&msecTotal1, start1, stop1);
	printf(" BatchNormalize: %f ms\n ", msecTotal1);
#endif
#ifdef _DEBUG
	int dwSize = top_data_size[0].data_dim[1] * top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3];
	float *pfDataOut = new float[dwSize];
	cudaMemcpy(pfDataOut, output_data_map[0]->m_gpu.pfData_gpu, dwSize * sizeof(float), cudaMemcpyDeviceToHost);
	delete[] pfDataOut;
	cudaDeviceSynchronize();
	printf("BatchNormalize:%s\n", cudaGetErrorString(cudaGetLastError()));
#endif
	return CUDA_RETURN_VALUE;
}
