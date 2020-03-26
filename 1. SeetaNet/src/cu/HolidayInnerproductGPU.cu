#include "HolidayInnerproductGPU.h"

HolidayInnerProductGPU::HolidayInnerProductGPU()
{
	pfParam_d = 0;
	pfBias_d = 0;
}
HolidayInnerProductGPU::~HolidayInnerProductGPU()
{
}
//MatrixMult
__global__ static void gInnerProduct_kernel(float *pfA, float *pfB, float *pfC, int dwN, int dwM, int dwP)
{
	__shared__ float pfTmpA[TRANS_BLOCK_DIM][TRANS_BLOCK_DIM];
	__shared__ float pfTmpB[TRANS_BLOCK_DIM][TRANS_BLOCK_DIM];
	int dwGlobalIdxN = blockDim.y * blockIdx.y + threadIdx.y;
	int dwGlobalIdxM = blockDim.x * blockIdx.x + threadIdx.x;
	int dwLocalIdxN = threadIdx.y;
	int dwLocalIdxM = threadIdx.x;
	float fResults = 0;
	float fComp = 0;
	for (int j = 0; j < dwP; j += TRANS_BLOCK_DIM)
	{
		if (dwGlobalIdxN < dwN && dwLocalIdxM + j < dwP)
		{
			pfTmpA[dwLocalIdxN][dwLocalIdxM] = pfA[dwGlobalIdxN * dwP + dwLocalIdxM + j];
		}
		else
		{
			pfTmpA[dwLocalIdxN][dwLocalIdxM] = 0;
		}

		if (dwGlobalIdxM < dwM && dwLocalIdxN + j < dwP)
		{
			pfTmpB[dwLocalIdxN][dwLocalIdxM] = pfB[(dwLocalIdxN + j) * dwM + dwGlobalIdxM];
		}
		else
		{
			pfTmpB[dwLocalIdxN][dwLocalIdxM] = 0;
		}
		__syncthreads();
		for (int i = 0; i < TRANS_BLOCK_DIM; i++)
		{
			float fTmp;
			fComp -= pfTmpA[dwLocalIdxN][i] * pfTmpB[i][dwLocalIdxM];
			fTmp = fResults - fComp;
			fComp = (fTmp - fResults) + fComp;
			fResults = fTmp;
		}
		__syncthreads();
	}

	if (dwGlobalIdxM < dwM && dwGlobalIdxN < dwN)
	{
		pfC[dwGlobalIdxN * dwM + dwGlobalIdxM] += fResults;
	}

}
__global__ static void gTransParam_kernel(float *pfParamIn, float *pfParamOut, int dwNum, int dwRows, int dwCols, int dwSlices)
{
	int dwIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (dwIdx < dwNum * dwSlices * dwCols * dwRows)
	{
		int dwDimN = dwIdx / (dwSlices * dwRows * dwCols);
		int dwDimR = (dwIdx - dwDimN * (dwSlices * dwRows * dwCols)) / (dwSlices * dwCols);
		int dwDimC = (dwIdx - dwDimN * (dwSlices * dwRows * dwCols) - dwDimR * (dwSlices * dwCols)) / dwSlices;
		int dwDimS = dwIdx - dwDimN * (dwSlices * dwRows * dwCols) - dwDimR * (dwSlices * dwCols) - dwDimC * dwSlices;
		pfParamOut[dwIdx] = pfParamIn[dwDimS * (dwNum * dwRows * dwCols) + dwDimR * (dwNum * dwCols) + dwDimC * dwNum + dwDimN];
	}
}
__global__ static void gBiasTrans_kernel(float *pfBias, float *pfOut, int dwCol, int dwRow, int dwSlice)
{
	int dwIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int dwDimS = dwIdx / (dwRow * dwCol);
	int dwDimR = (dwIdx - dwDimS * (dwRow * dwCol)) / dwCol;
	int dwDimC = dwIdx - dwDimS * (dwRow * dwCol) - dwDimR * dwCol;
	if (dwIdx < dwCol * dwRow * dwSlice)
		pfOut[dwIdx] = pfBias[dwDimS + dwDimC * dwSlice + dwDimR * dwSlice * dwCol];
}
int HolidayInnerProductGPU::Init(Holiday_LayerParameter& inputparam, HolidayNetResource<float> *pNetResource)
{
	pNetResourceGpu = (HolidayNetResourceGpu *)pNetResource->pNetResourceGpu;

	int index = inputparam.bottom_index(0);
	bottom_data_size.resize(1);
	bottom_data_size[0] = pNetResource->feature_vector_size[index];
	const Holiday_BlobProto& inner_param = inputparam.inner_product_param().inner_param();
	
	top_data_size.resize(1);
	top_data_size[0].data_dim.resize(4);
	top_data_size[0].data_dim[0] = pNetResource->max_batch_size;
	top_data_size[0].data_dim[2] = 1;
	top_data_size[0].data_dim[3] = 1;
	this->top_data_size[0].data_dim[1] = inner_param.shape().dim(0);
	//int dwSize1 = top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3] * top_data_size[0].data_dim[1];
	int dwSize = top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3] * top_data_size[0].data_dim[1];

	const float *innerproduct_value = inputparam.mutable_inner_product_param()->mutable_inner_param()->mutable_data()->mutable_data();
	int n_element = inputparam.mutable_inner_product_param()->mutable_inner_param()->mutable_data()->size();

	std::vector<int> inner_param_shape;
	inner_param_shape.clear();
	for (size_t i = 0; i < inputparam.mutable_inner_product_param()->mutable_inner_param()->shape().dim_size(); i++)
	{
		inner_param_shape.push_back(inputparam.inner_product_param().inner_param().shape().dim(i));
	}

	/*float* tmp_float_inner_param = new float[n_element];
	for (size_t i = 0; i < inner_param_shape[1]; i++)
	{
		for (size_t j = 0; j < inner_param_shape[0]; j++)
		{
			tmp_float_inner_param[i*inner_param_shape[0] + j] = innerproduct_value[j*inner_param_shape[1]+ i];
		}
	}*/

	CUDA_ERROR(SafeCudaMalloc((void **)&pfParam_d, n_element * sizeof(float)));
	cudaMemcpyAsync(pfParam_d, innerproduct_value, n_element * sizeof(float), cudaMemcpyHostToDevice, pNetResourceGpu->main_stream);

	//delete[]tmp_float_inner_param;

	const float *bias_value = inputparam.mutable_inner_product_param()->mutable_bias_param()->mutable_data()->mutable_data();
	int bias_value_size = inputparam.mutable_inner_product_param()->mutable_bias_param()->mutable_data()->size();

	CUDA_ERROR(SafeCudaMalloc((void **)&pfBias_d, bias_value_size * pNetResourceGpu->dwMaxBatchNum * sizeof(float)));
	float *pfBiasTmp;
	CUDA_ERROR(SafeCudaMalloc((void **)&pfBiasTmp, bias_value_size * sizeof(float)));
	CUDA_ERROR(cudaMemcpyAsync(pfBiasTmp, bias_value, bias_value_size * sizeof(float), cudaMemcpyHostToDevice, pNetResourceGpu->main_stream));
	for (int i = 0; i < pNetResourceGpu->dwMaxBatchNum; ++i)
	{
		gBiasTrans_kernel << <CUDA_BLOCK(bias_value_size, CUDA_THREAD_NUM), CUDA_THREAD_NUM, 0, pNetResourceGpu->main_stream>> >(pfBiasTmp, pfBias_d + i * bias_value_size,
			top_data_size[0].data_dim[3], top_data_size[0].data_dim[2], top_data_size[0].data_dim[1]);
	}
	CUDA_ERROR(cudaFree(pfBiasTmp));

	cudaStreamSynchronize(pNetResourceGpu->main_stream);
	return CUDA_RETURN_VALUE;
}
int HolidayInnerProductGPU::Exit()
{
	if (pfParam_d) cudaFree(pfParam_d);
	if (pfBias_d) cudaFree(pfBias_d);
	return CUDA_RETURN_VALUE;
}

int HolidayInnerProductGPU::Process(std::vector<HolidayFeatureMap<float>*> bottom_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map)
{
#ifdef _DEBUG
	cudaEvent_t start1;
	cudaEventCreate(&start1);
	cudaEvent_t stop1;
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, NULL);
#endif
	bottom_data_map[0]->m_gpu.Gpu_DataIn(pNetResourceGpu, bottom_data_map[0]->dwStorageType, bottom_data_map[0]->m_cpu.dataMemoryPtr());
	int dwSizeout = top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3] * top_data_size[0].data_dim[1];
	int dwSizein = bottom_data_size[0].data_dim[2] * bottom_data_size[0].data_dim[3] * bottom_data_size[0].data_dim[1];
	CUDA_ERROR(cudaMemcpyAsync(output_data_map[0]->m_gpu.pfData_gpu, pfBias_d, dwSizeout * bottom_data_map[0]->m_gpu.shape_[0] * sizeof(float), cudaMemcpyDeviceToDevice, pNetResourceGpu->main_stream));
#ifdef _X64_
	float fAlpha = 1.f;
	float fBeta = 1.f;
	CUBLAS_ERROR(cublasSgemm(pNetResourceGpu->Handle_cublas, CUBLAS_OP_N, CUBLAS_OP_N, dwSizeout,
		bottom_data_map[0]->m_gpu.shape_[0], dwSizein, &fAlpha,
		pfParam_d, dwSizeout, bottom_data_map[0]->m_gpu.pfData_gpu, dwSizein,
		&fBeta, output_data_map[0]->m_gpu.pfData_gpu, dwSizeout));

#else
	int dwSize2 = top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3] * top_data_size[0].data_dim[1];

	dim3 blocksize(CUDA_BLOCK(dwSize2, TRANS_BLOCK_DIM), CUDA_BLOCK(bottom_data_map[0]->m_gpu.shape_[0], TRANS_BLOCK_DIM));
	dim3 threadsize(TRANS_BLOCK_DIM, TRANS_BLOCK_DIM);
	gInnerProduct_kernel << <blocksize, threadsize, 0, pNetResourceGpu->main_stream>> >(bottom_data_map[0]->m_gpu.pfData_gpu, pfParam_d,
		output_data_map[0]->m_gpu.pfData_gpu, bottom_data_map[0]->m_gpu.shape_[0], dwSizeout, dwSizein);
#endif
	output_data_map[0]->dwStorageType = DATA_GPU;
	output_data_map[0]->m_gpu.shape_[0] = bottom_data_map[0]->m_gpu.shape_[0];
	//output_data_map[0]->shape_[0] = output_data_map[0]->m_gpu.shape_[0];
	output_data_map[0]->data_shape[0] = bottom_data_map[0]->data_shape[0];
	output_data_map[0]->data_shape[1] = top_data_size[0].data_dim[1];
	output_data_map[0]->data_shape[2] = top_data_size[0].data_dim[2];
	output_data_map[0]->data_shape[3] = top_data_size[0].data_dim[3];

#ifdef _DEBUG
	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
	float msecTotal1 = 0.0f;
	cudaEventElapsedTime(&msecTotal1, start1, stop1);
	printf(" InnerProduct: %f ms \n", msecTotal1);
#endif
#ifdef _DEBUG
	int dwSize3 = top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3] * top_data_size[0].data_dim[1];
	float *pfDataOut = new float[dwSize3];
	cudaMemcpy(pfDataOut, output_data_map[0]->m_gpu.pfData_gpu, dwSize3 * sizeof(float), cudaMemcpyDeviceToHost);
	delete[] pfDataOut;
	cudaDeviceSynchronize();
	printf("InnerProduct:%s\n", cudaGetErrorString(cudaGetLastError()));
#endif
	return CUDA_RETURN_VALUE;
}
