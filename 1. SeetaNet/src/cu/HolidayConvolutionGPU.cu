#include"HolidayConvolutionGPU.h"
#include"HolidayNetResource.h"
#include"HolidayCNN_proto.pb.h"
#include"HolidayBlobGpu.h"

HolidayConvolutionGPU::HolidayConvolutionGPU()
	: pfKernel_d(nullptr), pfBias_d(nullptr), ppfBlas(nullptr), ppfBlas_d(nullptr), ppfKernel_d(nullptr)
{

}
HolidayConvolutionGPU::~HolidayConvolutionGPU()
{

}

__global__ static void gConvMatrixMult_kernel(float *pfA, float *pfB, float *pfC, int dwN, int dwM, int dwP, int dwG)
{
	__shared__ float pfTmpA[16][16];
	__shared__ float pfTmpB[16][16];

	int dwDimNG = (blockDim.y * blockIdx.y + threadIdx.y) / (CUDA_BLOCK(dwN, 16) * 16);
	int dwDimG = dwDimNG % dwG;
	float *pfOffA = pfA + dwDimG * dwN * dwP;
	float *pfOffB = pfB + dwDimNG * dwP * dwM;
	float *pfOffC = pfC + dwDimNG * dwN * dwM;

	int dwGlobalIdxN = (blockDim.y * blockIdx.y + threadIdx.y) % (CUDA_BLOCK(dwN, 16) * 16);
	int dwGlobalIdxM = blockDim.x * blockIdx.x + threadIdx.x;
	int dwLocalIdxN = threadIdx.y;
	int dwLocalIdxM = threadIdx.x;

	float fResults = 0;
	float fComp = 0;
	for (int j = 0; j < dwP; j += 16)
	{
		if (dwGlobalIdxN < dwN && dwLocalIdxM + j < dwP)
		{
			pfTmpA[dwLocalIdxN][dwLocalIdxM] = pfOffA[dwGlobalIdxN * dwP + dwLocalIdxM + j];
		}
		else
		{
			pfTmpA[dwLocalIdxN][dwLocalIdxM] = 0;
		}

		if (dwGlobalIdxM < dwM && dwLocalIdxN + j < dwP)
		{
			pfTmpB[dwLocalIdxN][dwLocalIdxM] = pfOffB[(dwLocalIdxN + j) * dwM + dwGlobalIdxM];
		}
		else
		{
			pfTmpB[dwLocalIdxN][dwLocalIdxM] = 0;
		}
		__syncthreads();
		for (int i = 0; i < 16; i++)
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
		pfOffC[dwGlobalIdxN * dwM + dwGlobalIdxM] += fResults;
	}

}

__global__ static void gInputTrans_kernel(float *pfDataIn, float *pfDataOut, int dwSize, int dwRowIn, int dwColIn,
	int dwSliceIn, int dwRowOut, int dwColOut, int dwSliceOut, int dwStrideH, int dwStrideW,
	int dwPadH, int dwPadW,
    int dwShiftH, int dwShiftW,
    int dwDilationH, int dwDilationW, int dwKernelH, int dwKernelW)
{
    dwPadH += dwShiftH;
    dwPadW += dwShiftW;
	int dwIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (dwIdx < dwSize)
	{
		int dwDimN = dwIdx / (dwSliceOut * dwRowOut * dwColOut);
		int dwDim2S = dwIdx % (dwSliceOut * dwRowOut * dwColOut) / (dwRowOut * dwColOut);
		int dwDim2R = dwIdx % (dwRowOut * dwColOut) / dwColOut;
		int dwDim2C = dwIdx % dwColOut;
		int dwDim1R = dwDim2R * dwStrideH - dwPadH;
		int dwDim1C = dwDim2C * dwStrideW - dwPadW;
		int dwDim1S = dwDim2S;
		int dwIdxOut = ((dwDimN * dwSliceOut + dwDim1S) * dwKernelH * dwKernelW * dwRowOut + dwDim2R) * dwColOut + dwDim2C;
		int dwIdxIn = dwDim1C + dwColIn * (dwDim1R + dwRowIn * (dwDim1S + dwSliceIn * dwDimN));
		for (int i = 0; i < dwKernelH; i++)
		{
			for (int j = 0; j < dwKernelW; j++)
			{
				if (dwDim1R + i * dwDilationH >= 0 && dwDim1R + i * dwDilationH < dwRowIn
					&& dwDim1C + j * dwDilationW >= 0 && dwDim1C + j * dwDilationW < dwColIn)
				{
					pfDataOut[dwIdxOut + dwColOut * dwRowOut * (i * dwKernelW + j)] =
						pfDataIn[dwIdxIn + j * dwDilationW + dwColIn * i * dwDilationH];
				}
				else
				{
					pfDataOut[dwIdxOut + dwColOut * dwRowOut * (i * dwKernelW + j)] = 0;
				}
			}
		}
	}
}

//__global__ static void gBiasSet_kernel(float *pfBias, float *pfOut, int dwBiasStep, int dwOutSize)
//{
//	int dwIdx = threadIdx.x + blockIdx.x * blockDim.x;
//	if (dwIdx < dwOutSize)
//		pfOut[dwIdx] = pfBias[dwIdx / dwBiasStep];
//}

__global__ static void gBiasSet_kernel(float *pfBias, float *pfOut, int dwBiasStep, int dwExtStep, int dwOutSize)
{
	int dwIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (dwIdx < dwOutSize)
		pfOut[dwIdx] = pfBias[(dwIdx % dwExtStep) / dwBiasStep];
}


int HolidayConvolutionGPU::Init(Holiday_LayerParameter &inputparam, HolidayNetResource<float> *p_holiday_net_resource)
{
	pNetResourceGpu = (HolidayNetResourceGpu *)p_holiday_net_resource->pNetResourceGpu;
//
	dwKernelNum = inputparam.convolution_param().kernel_param().shape().dim(0);
	dwKernelRows = inputparam.convolution_param().kernel_param().shape().dim(2);
	dwKernelCols = inputparam.convolution_param().kernel_param().shape().dim(3);
	dwKernelSlices = inputparam.convolution_param().kernel_param().shape().dim(1);
	
	//std::cout << "conv init 1" << std::endl;

	int bottom_index = inputparam.bottom_index(0);
	HolidayDataSize bottom_size = p_holiday_net_resource->feature_vector_size[bottom_index];
	this->bottom_data_size.resize(1);
	this->bottom_data_size[0] = bottom_size;
	
	std::vector<int> shape;
	const ::Holiday_BlobShape& tmp_shape = inputparam.convolution_param().kernel_param().shape();

	//std::cout << "conv init 2" << std::endl;

	for (int i = 0; i < tmp_shape.dim_size(); i++)
	{
		shape.push_back(tmp_shape.dim(i));
	}
	dwGroup = bottom_data_size[0].data_dim[1] / dwKernelSlices;
	dwStrideH = inputparam.convolution_param().stride_height();
	dwStrideW = inputparam.convolution_param().stride_width();
	dwPadH = inputparam.convolution_param().pad_height();
	dwPadW = inputparam.convolution_param().pad_width();
	dwDilationH = inputparam.convolution_param().dilation_height();
	dwDilationW = inputparam.convolution_param().dilation_width();

	if (inputparam.convolution_param().has_tf_padding())
	{
		m_tf_padding = inputparam.convolution_param().tf_padding();
	}


	top_data_size.resize(1);
	// calculate top blobs
	Calculate(bottom_data_size[0].data_dim, top_data_size[0].data_dim);

	bool is_1x1_conv = dwKernelRows == 1 && dwKernelCols == 1 && dwPadH == 0 && dwPadW == 0 && dwStrideH == 1 && dwStrideW == 1;

	// tmp buffer
	int dwKernelSize = dwKernelSlices * dwKernelRows * dwKernelCols;
	if (!is_1x1_conv) {
		gTmpBuffer_gpu(pNetResourceGpu, top_data_size[0].data_dim[3] * top_data_size[0].data_dim[2] * dwKernelSize * dwGroup * pNetResourceGpu->dwMaxBatchNum * sizeof(float));
	}
	// transData
	ppfBlas = new float*[dwGroup * pNetResourceGpu->dwMaxBatchNum * 2];
	CUDA_ERROR(SafeCudaMalloc((void **)&ppfBlas_d, dwGroup * pNetResourceGpu->dwMaxBatchNum * 2 * sizeof(float*)));
	//kernel param
	CUDA_ERROR(SafeCudaMalloc((void **)&pfKernel_d, dwKernelSize * dwKernelNum * sizeof(float)));
	const float *pfKernelT = inputparam.mutable_convolution_param()->mutable_kernel_param()->mutable_data()->mutable_data();
	CUDA_ERROR(cudaMemcpyAsync(pfKernel_d, pfKernelT, dwKernelSize *dwKernelNum* sizeof(float), cudaMemcpyHostToDevice, pNetResourceGpu->main_stream));
	
	std::unique_ptr<float*[]> ppfKernel(new float*[dwGroup * pNetResourceGpu->dwMaxBatchNum]);
	CUDA_ERROR(SafeCudaMalloc((void **)&ppfKernel_d, dwGroup * pNetResourceGpu->dwMaxBatchNum * sizeof(float*)));
	for (int i = 0; i < dwGroup * pNetResourceGpu->dwMaxBatchNum; ++i)
	{
		ppfKernel[i] = pfKernel_d + (i % dwGroup) * dwKernelSize * (dwKernelNum / dwGroup);
	}
	CUDA_ERROR(cudaMemcpyAsync(ppfKernel_d, ppfKernel.get(), dwGroup * pNetResourceGpu->dwMaxBatchNum * sizeof(float *), cudaMemcpyHostToDevice, pNetResourceGpu->main_stream));
	//bias param
	int dwsize = top_data_size[0].data_dim[1] * top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3];
	CUDA_ERROR(SafeCudaMalloc((void **)&pfBias_d, dwKernelNum * sizeof(float)));
	//float *pfBiasTmp_d;
	//CUDA_ERROR(SafeCudaMalloc((void **)&pfBiasTmp_d, dwKernelNum * sizeof(float)));
	
	if (inputparam.convolution_param().bias_param().data().size())
	{
		const float *pfBias = inputparam.mutable_convolution_param()->mutable_bias_param()->mutable_data()->mutable_data();
		CUDA_ERROR(cudaMemcpyAsync(pfBias_d, pfBias, dwKernelNum * sizeof(float), cudaMemcpyHostToDevice, pNetResourceGpu->main_stream));
	}
	else
	{
		CUDA_ERROR(cudaMemsetAsync(pfBias_d, 0, dwKernelNum * sizeof(float), pNetResourceGpu->main_stream));
	}

	cudaStreamSynchronize(pNetResourceGpu->main_stream);
	return CUDA_RETURN_VALUE;
}
int HolidayConvolutionGPU::Exit()
{
	if (pfKernel_d) cudaFree(pfKernel_d);
	if (pfBias_d) cudaFree(pfBias_d);
	if (ppfBlas) delete[]ppfBlas;
	if (ppfBlas_d) cudaFree(ppfBlas_d);
	if (ppfKernel_d) cudaFree(ppfKernel_d);
	return CUDA_RETURN_VALUE;
}

int HolidayConvolutionGPU::Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map)
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

	Calculate(input_data_map[0]->data_shape, output_data_map[0]->data_shape);

	output_data_map[0]->m_gpu.shape_ = output_data_map[0]->data_shape;
	output_data_map[0]->m_gpu.data_size = output_data_map[0]->data_shape[0] * output_data_map[0]->data_shape[1] * output_data_map[0]->data_shape[2] * output_data_map[0]->data_shape[3];


	
	
	//gInputTrans_kernel << <CUDA_BLOCK(top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3]
	//	* bottom_data_size[0].data_dim[1] * input_data_map[0]->m_gpu.shape_[0], CUDA_THREAD_NUM), CUDA_THREAD_NUM, 0, pNetResourceGpu->main_stream>> >
	//	(input_data_map[0]->m_gpu.pfData_gpu, pfDataTrans_d,
	//	top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3] * bottom_data_size[0].data_dim[1] * input_data_map[0]->m_gpu.shape_[0],
	//	bottom_data_size[0].data_dim[2], bottom_data_size[0].data_dim[3], bottom_data_size[0].data_dim[1],
	//	top_data_size[0].data_dim[2], top_data_size[0].data_dim[3], bottom_data_size[0].data_dim[1],
	//	dwStrideH, dwStrideW, dwPadH, dwPadW, dwDilationH, dwDilationW, dwKernelRows, dwKernelCols);
	int dwsize = output_data_map[0]->data_shape[3] * output_data_map[0]->data_shape[2] * output_data_map[0]->data_shape[1];

	int put_param = output_data_map[0]->data_shape[2] * output_data_map[0]->data_shape[3] * input_data_map[0]->data_shape[1] * input_data_map[0]->data_shape[0];

	bool is_1x1_conv = dwKernelRows == 1 && dwKernelCols == 1 && dwPadH == 0 && dwPadW == 0 && dwStrideH == 1 && dwStrideW == 1;

	float *pfDataTrans_d = nullptr;

	if (is_1x1_conv)
	{
		pfDataTrans_d = (float *)input_data_map[0]->m_gpu.pfData_gpu;
	}
	else
	{
		gTmpBuffer_gpu(pNetResourceGpu, output_data_map[0]->m_gpu.data_size * dwKernelRows * dwKernelCols * dwGroup * sizeof(float));
		pfDataTrans_d = (float *)pNetResourceGpu->pubyConvTmpBuffer;
		gInputTrans_kernel << <CUDA_BLOCK(put_param, CUDA_THREAD_NUM), CUDA_THREAD_NUM, 0, pNetResourceGpu->main_stream>> >
			((float *)input_data_map[0]->m_gpu.pfData_gpu, pfDataTrans_d,
			output_data_map[0]->data_shape[2] * output_data_map[0]->data_shape[3] * input_data_map[0]->data_shape[1] * input_data_map[0]->data_shape[0],
			input_data_map[0]->data_shape[2], input_data_map[0]->data_shape[3], input_data_map[0]->data_shape[1],
			output_data_map[0]->data_shape[2], output_data_map[0]->data_shape[3], input_data_map[0]->data_shape[1],
			dwStrideH, dwStrideW,
	        dwPadH + m_tf_fake_padding_h, dwPadW + m_tf_fake_padding_w,
	        m_tf_conv_shift_h, m_tf_conv_shift_w,
	        dwDilationH, dwDilationW, dwKernelRows, dwKernelCols);
	}

#ifdef _DEBUG
	// int buffer_size = output_data_map[0]->m_gpu.data_size * dwKernelRows * dwKernelCols * dwGroup;
 //
	// float *pfcol_DataOut = new float[buffer_size];
	// CUDA_ERROR(cudaMemcpy(pfcol_DataOut, pfDataTrans_d, buffer_size* sizeof(float), cudaMemcpyDeviceToHost));
	// delete[] pfcol_DataOut;
	// cudaDeviceSynchronize();
	
#endif

	//gBiasSet_kernel << <CUDA_BLOCK(output_data_map[0]->m_gpu.shape_, CUDA_THREAD_NUM), CUDA_THREAD_NUM, 0, pNetResourceGpu->main_stream>> >(pfBias_d, (float *)output_data_map[0]->m_gpu.pfData_gpu,
	//	output_data_map[0]->data_shape[3] * output_data_map[0]->data_shape[2], output_data_map[0]->data_shape[3] * output_data_map[0]->data_shape[2] * output_data_map[0]->data_shape[1], output_data_map[0]->m_gpu.data_size);
	gBiasSet_kernel << <CUDA_BLOCK(output_data_map[0]->m_gpu.data_size, CUDA_THREAD_NUM), CUDA_THREAD_NUM, 0, pNetResourceGpu->main_stream>> >(pfBias_d,
		(float *)output_data_map[0]->m_gpu.pfData_gpu,
		output_data_map[0]->data_shape[3] * output_data_map[0]->data_shape[2], 
		output_data_map[0]->data_shape[3] * output_data_map[0]->data_shape[2] * output_data_map[0]->data_shape[1], 
		output_data_map[0]->m_gpu.data_size);

#ifdef _X64_

	if (dwsize / dwGroup < 10000)
	{
		for (int i = 0; i < dwGroup * input_data_map[0]->m_gpu.shape_[0]; i++)
		{
			ppfBlas[i] = pfDataTrans_d + i * dwKernelSlices * dwKernelRows * dwKernelCols * top_data_size[0].data_dim[3] * top_data_size[0].data_dim[2];
			ppfBlas[i + dwGroup * input_data_map[0]->m_gpu.shape_[0]] = output_data_map[0]->m_gpu.pfData_gpu + i * dwsize / dwGroup;
		}
		CUDA_ERROR(cudaMemcpyAsync(ppfBlas_d, ppfBlas, dwGroup * input_data_map[0]->m_gpu.shape_[0] * 2 * sizeof(float *), cudaMemcpyHostToDevice, pNetResourceGpu->main_stream));
		//CUDA_ERROR(cudaMemcpy(output_data_map[0]->m_gpu.pfData_gpu, pfBias_d, dwsize * input_data_map[0]->m_gpu.shape_[0] * sizeof(float), cudaMemcpyDeviceToDevice));
		float fAlpha = 1.f;
		float fBeta = 1.f;
		
		CUBLAS_ERROR(cublasSgemmBatched(pNetResourceGpu->Handle_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
			output_data_map[0]->data_shape[3] * output_data_map[0]->data_shape[2], output_data_map[0]->data_shape[1] / dwGroup,
			dwKernelSlices * dwKernelRows * dwKernelCols, &fAlpha,
			(const float **)ppfBlas_d, output_data_map[0]->data_shape[3] * output_data_map[0]->data_shape[2],
			(const float **)ppfKernel_d, dwKernelSlices * dwKernelRows * dwKernelCols, &fBeta,
			ppfBlas_d + dwGroup * input_data_map[0]->data_shape[0], output_data_map[0]->data_shape[3] * output_data_map[0]->data_shape[2],
			dwGroup * input_data_map[0]->data_shape[0]));

	}
	else
	{
		//CUDA_ERROR(cudaMemcpy(output_data_map[0]->m_gpu.pfData_gpu, pfBias_d, dwsize * input_data_map[0]->m_gpu.shape_[0] * sizeof(float), cudaMemcpyDeviceToDevice));
		for (int i = 0; i < dwGroup * input_data_map[0]->m_gpu.shape_[0]; i++)
		{
			float fAlpha = 1.f;
			float fBeta = 1.f;
			
			int blas_n = output_data_map[0]->data_shape[3] * output_data_map[0]->data_shape[2];
			int blas_m = output_data_map[0]->data_shape[1] / dwGroup;
			int blas_k = dwKernelSlices * dwKernelRows * dwKernelCols;
			CUBLAS_ERROR(cublasSgemm(pNetResourceGpu->Handle_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
				blas_n, blas_m,
				blas_k, &fAlpha,
				pfDataTrans_d + i * dwKernelSlices * dwKernelRows * dwKernelCols * output_data_map[0]->data_shape[3] * output_data_map[0]->data_shape[2],
				blas_n,
				pfKernel_d + (i % dwGroup) * dwKernelSlices * dwKernelRows * dwKernelCols * (dwKernelNum / dwGroup), blas_k,
				&fBeta,
				output_data_map[0]->m_gpu.pfData_gpu + i * output_data_map[0]->data_shape[3] * output_data_map[0]->data_shape[2] * output_data_map[0]->data_shape[1] / dwGroup,
				blas_n));
		}
	}
#else
	// CUDA_ERROR(cudaMemcpy(output_data_map[0]->m_gpu.pfData_gpu, pfBias_d, dwsize * input_data_map[0]->m_gpu.shape_[0] * sizeof(float), cudaMemcpyDeviceToDevice));
	int dwN = output_data_map[0]->data_shape[1] / dwGroup;
	int dwM = output_data_map[0]->data_shape[3] * output_data_map[0]->data_shape[2];
	int dwP = dwKernelSlices * dwKernelRows * dwKernelCols;
	dim3 blocksize(CUDA_BLOCK(dwM, 16), CUDA_BLOCK(dwN, 16) * dwGroup * input_data_map[0]->m_gpu.shape_[0]);
	dim3 threadsize(16, 16);
	gConvMatrixMult_kernel << <blocksize, threadsize, 0, pNetResourceGpu->main_stream>> >(pfKernel_d, pfDataTrans_d, output_data_map[0]->m_gpu.pfData_gpu, dwN, dwM, dwP, dwGroup);
#endif
	output_data_map[0]->dwStorageType = DATA_GPU;
	

#ifdef _DEBUG
	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
	float msecTotal1 = 0.0f;
	cudaEventElapsedTime(&msecTotal1, start1, stop1);
	//printf(" Convolution: %f ms [%d < 10000 : batch ? stream]\n ", msecTotal1, top_data_size[0].dwSize / dwGroup);
#endif
#ifdef _DEBUG
	float *pfDataOut = new float[output_data_map[0]->m_gpu.data_size];
	output_data_map[0]->m_gpu.Gpu_DataOut(pNetResourceGpu, DATA_CPU_WIDTH, pfDataOut);
	delete[] pfDataOut;
	cudaDeviceSynchronize();
	printf("Convolution:%s\n", cudaGetErrorString(cudaGetLastError()));
#endif
	return CUDA_RETURN_VALUE;
}

int HolidayConvolutionGPU::Caculate(const int height, const int width, const int kernel_h, const int kernel_w,
									   const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int dilation_h, const int dilation_w,
									   int& output_h, int& output_w)
{
	if (m_tf_padding == "VALID")
	{
		output_h = ceil((height + 2 * pad_h -
						 (dilation_h * (kernel_h - 1))) / float(stride_h));
		output_w = ceil((width + 2 * pad_w -
						 (dilation_w * (kernel_w - 1))) / float(stride_w));
	}
	else if (m_tf_padding == "SAME")
	{
		output_h = ceil((height + 2 * pad_h) / float(stride_h));
		output_w = ceil((width + 2 * pad_w) / float(stride_w));

		int original_view_h = height + 2 * pad_h;
		int original_view_w = width + 2 * pad_w;

		int need_view_h = output_h * stride_h + kernel_h - 1;
		int need_view_w = output_w * stride_w + kernel_w - 1;

		m_tf_fake_padding_h = (need_view_h - original_view_h) / 2;
        m_tf_fake_padding_w = (need_view_w - original_view_w) / 2;

        int tf_need_view_h = (output_h - 1) * stride_h + kernel_h;
        int tf_need_view_w = (output_w - 1) * stride_w + kernel_w;

        m_tf_conv_shift_h = -m_tf_fake_padding_h + (tf_need_view_h - original_view_h) / 2;
        m_tf_conv_shift_w = -m_tf_fake_padding_w + (tf_need_view_w - original_view_w) / 2;
	}
	else
	{
		output_h = (height + 2 * pad_h -
					(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
		output_w = (width + 2 * pad_w -
					(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	}

	return 0;
}

int HolidayConvolutionGPU::Calculate(const std::vector<int> &bottom_shape, std::vector<int> &top_shape) {
	top_shape.resize(4);
	top_shape[0] = bottom_shape[0];
	top_shape[1] = dwKernelNum;
	Caculate(bottom_shape[2], bottom_shape[3],
			 dwKernelRows, dwKernelCols,
			 dwPadH, dwPadW,
			 dwStrideH, dwStrideW,
			 dwDilationH, dwDilationW,
			 top_shape[2], top_shape[3]);
	return 0;
}



