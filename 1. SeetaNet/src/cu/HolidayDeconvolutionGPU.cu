#include "HolidayDeconvolutionGPU.h"

#include "HolidayCommonCuda.h"



HolidayDeconvolutionGPU::HolidayDeconvolutionGPU()
{
    pfWeight_d = 0;
    pfBias_d = 0;
    ppfBlas = 0;
    ppfBlas_d = 0;
    ppfWeight_d = 0;
}
HolidayDeconvolutionGPU::~HolidayDeconvolutionGPU()
{
}
__global__ static void gDeconvMatrixMult_kernel(float *pfA, float *pfB, float *pfC, int dwN, int dwM, int dwP, int dwG)
{
    __shared__ float pfTmpA[TRANS_BLOCK_DIM][TRANS_BLOCK_DIM];
    __shared__ float pfTmpB[TRANS_BLOCK_DIM][TRANS_BLOCK_DIM];

    int dwDimNG = (blockDim.y * blockIdx.y + threadIdx.y) / (CUDA_BLOCK(dwN, TRANS_BLOCK_DIM) * TRANS_BLOCK_DIM);
    int dwDimG = dwDimNG % dwG;
    float *pfOffA = pfA + dwDimG * dwN * dwP;
    float *pfOffB = pfB + dwDimNG * dwP * dwM;
    float *pfOffC = pfC + dwDimNG * dwN * dwM;

    int dwGlobalIdxN = (blockDim.y * blockIdx.y + threadIdx.y) % (CUDA_BLOCK(dwN, TRANS_BLOCK_DIM) * TRANS_BLOCK_DIM);
    int dwGlobalIdxM = blockDim.x * blockIdx.x + threadIdx.x;
    int dwLocalIdxN = threadIdx.y;
    int dwLocalIdxM = threadIdx.x;

    float fResults = 0;
    float fComp = 0;
    for (int j = 0; j < dwP; j += TRANS_BLOCK_DIM)
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
        pfOffC[dwGlobalIdxN * dwM + dwGlobalIdxM] += fResults;
    }

}

// true, false
// C = C * SC + A * B
// A for (P, N) memory layout // for trans
// B for (P, M) memory layout // for no trans

__global__ static void gInnerProduct_TF_kernel(const float *pfA, const float *pfB, float *pfC, int dwN, int dwM, int dwP, float fSC = 0)
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
			pfTmpA[dwLocalIdxN][dwLocalIdxM] = pfA[(dwLocalIdxM + j) * dwN + dwGlobalIdxN]; //pfA[dwGlobalIdxN * dwP + dwLocalIdxM + j];
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
		pfC[dwGlobalIdxN * dwM + dwGlobalIdxM] *= fSC;
		pfC[dwGlobalIdxN * dwM + dwGlobalIdxM] += fResults;
	}

}

// false, false
// C = A * B
// A for (N, P) memory layout // for no trans
// B for (P, M) memory layout // for no trans

__global__ static void gInnerProduct_FF_kernel(const float *pfA, const float *pfB, float *pfC, int dwN, int dwM, int dwP, float fSC = 0)
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
		pfC[dwGlobalIdxN * dwM + dwGlobalIdxM] *= fSC;
		pfC[dwGlobalIdxN * dwM + dwGlobalIdxM] += fResults;
	}

}

__global__ static void gBiasAdd_kernel(float *pfBias, float *pfOut, int dwBiasStep, int dwExtStep, int dwOutSize)
{
	int dwIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (dwIdx < dwOutSize)
		pfOut[dwIdx] += pfBias[(dwIdx % dwExtStep) / dwBiasStep];
}

__global__ static void gOutputTrans_kernel(float *pfDataIn, float *pfDataOut, int dwSize, int dwRowIn, int dwColIn,
    int dwSliceIn, int dwRowOut, int dwColOut, int dwSliceOut, int dwStrideH, int dwStrideW,
    int dwPadH, int dwPadW, int dwDilationH, int dwDilationW, int dwKernelH, int dwKernelW)
{
    int dwIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (dwIdx < dwSize)
    {
        int dwDimN = dwIdx / (dwSliceOut * dwRowOut * dwColOut);
        int dwDim2S = dwIdx % (dwSliceOut * dwRowOut * dwColOut) / (dwRowOut * dwColOut);
        int dwDim2R = dwIdx % (dwRowOut * dwColOut) / dwColOut;
        int dwDim2C = dwIdx % dwColOut;
        int dwKernelExW = (dwKernelW - 1) * dwDilationW + 1;
        int dwKernelExH = (dwKernelH - 1) * dwDilationH + 1;
        int dwDimKS = 0;
        int fSum = 0.f;
        for (int i = 0; i < dwKernelExH; i += dwDilationH)
        {
            for (int j = 0; j < dwKernelExW; j += dwDilationW, dwDimKS++)
            {
                int dwDim1R = (dwDim2R - i + dwPadH) / dwStrideH;
                int dwDim1C = (dwDim2C - j + dwPadW) / dwStrideW;
                if ((dwDim2R - i + dwPadH) % dwStrideH == 0 && dwDim1R >= 0 && dwDim1R < dwRowIn
                    && (dwDim2C - j + dwPadW) % dwStrideW == 0 && dwDim1C >= 0 && dwDim1C < dwColIn)
                {
                    fSum += pfDataIn[(((dwDimN * dwSliceIn + dwDim2S) * dwKernelH * dwKernelW + dwDimKS) * dwRowIn + dwDim1R) * dwColIn + dwDim1C];
                }
            }
        }
        pfDataOut[dwIdx] = fSum;
    }
}
__global__ static void gBiasSet_kernel(float *pfBias, float *pfOut, int dwBiasStep, int dwExtStep, int dwOutSize)
{
    int dwIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (dwIdx < dwOutSize)
        pfOut[dwIdx] = pfBias[(dwIdx % dwExtStep) / dwBiasStep];
}
int HolidayDeconvolutionGPU::Init(Holiday_LayerParameter &inputparam, HolidayNetResource<float> *p_holiday_net_resource)
{
    pNetResourceGpu = (HolidayNetResourceGpu *)p_holiday_net_resource->pNetResourceGpu;
    dwKernelNum = inputparam.convolution_param().kernel_param().shape().dim(0);
    dwKernelRows = inputparam.convolution_param().kernel_param().shape().dim(2);
    dwKernelCols = inputparam.convolution_param().kernel_param().shape().dim(3);
    dwKernelSlices = inputparam.convolution_param().kernel_param().shape().dim(1);
    //inputparam.convolution_param.

    dwWeightRows = inputparam.convolution_param().kernel_param().shape().dim(2);
    dwWeightCols = inputparam.convolution_param().kernel_param().shape().dim(3);
    dwKernelSlices = inputparam.convolution_param().kernel_param().shape().dim(1);

    int bottom_index = inputparam.bottom_index(0);
    HolidayDataSize bottom_size = p_holiday_net_resource->feature_vector_size[bottom_index];
    this->bottom_data_size.resize(1);
    this->bottom_data_size[0] = bottom_size;

    //VectorParam_gpu *pdjTmpParam = &inputparam.param_vector_gpu;
    dwStrideH = inputparam.convolution_param().stride_height();
    dwStrideW = inputparam.convolution_param().stride_width();
    dwPadH = inputparam.convolution_param().pad_height();
    dwPadW = inputparam.convolution_param().pad_width();
    dwDilationH = inputparam.convolution_param().dilation_height();
    dwDilationW = inputparam.convolution_param().dilation_width();
   
    dwGroup = inputparam.convolution_param().group();

    top_data_size.resize(1);
    top_data_size[0].data_dim.resize(4);
    top_data_size[0].data_dim[0] = p_holiday_net_resource->max_batch_size;
    top_data_size[0].data_dim[1] = dwKernelSlices;
    top_data_size[0].data_dim[2] = (bottom_data_size[0].data_dim[2] - 1)*dwStrideH - dwPadH * 2 + dwDilationH * (dwKernelRows - 1) + 1;
    top_data_size[0].data_dim[3] = (bottom_data_size[0].data_dim[3] - 1)*dwStrideW - dwPadW * 2 + dwDilationW * (dwKernelCols - 1) + 1;

    gTmpBuffer_gpu(pNetResourceGpu,
        this->bottom_data_size[0].data_dim[3] * this->bottom_data_size[0].data_dim[2] * dwKernelSlices* dwKernelRows* dwKernelCols * dwGroup * pNetResourceGpu->dwMaxBatchNum * sizeof(float));
    //ppfBlas = new float*[dwGroup * pNetResourceGpu->dwMaxBatchNum * 2];
    //CUDA_ERROR(SafeCudaMalloc((void **)&ppfBlas_d, dwGroup * pNetResourceGpu->dwMaxBatchNum * 2 * sizeof(float*)));

    CUDA_ERROR(SafeCudaMalloc((void **)&pfWeight_d, dwKernelNum * dwKernelSlices* dwKernelRows* dwKernelCols * sizeof(float)));
    CUDA_ERROR(cudaMemcpyAsync(pfWeight_d, inputparam.convolution_param().kernel_param().data().data(), dwKernelNum * dwKernelSlices* dwKernelRows* dwKernelCols * sizeof(float), cudaMemcpyHostToDevice, pNetResourceGpu->main_stream));
   
    if (inputparam.convolution_param().bias_param().data().size())
    {
        CUDA_ERROR(SafeCudaMalloc((void **)&pfBias_d, dwKernelSlices * sizeof(float)));
        CUDA_ERROR(cudaMemcpyAsync(pfBias_d, inputparam.convolution_param().bias_param().data().data(), dwKernelSlices * sizeof(float), cudaMemcpyHostToDevice, pNetResourceGpu->main_stream));
    }
    else
    {
        CUDA_ERROR(cudaMemsetAsync(pfBias_d, 0, dwKernelSlices * sizeof(float), pNetResourceGpu->main_stream));
    }
    //float *bias_multiplier_
    
    int tmp_bias_multiplier_length = top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3];
	std::unique_ptr<float[]> tmp_bias_multiplier(new float[tmp_bias_multiplier_length]);
    for (size_t i = 0; i < tmp_bias_multiplier_length; i++)
    {
        tmp_bias_multiplier[i] = 1.0f;
    }


    CUDA_ERROR(SafeCudaMalloc((void **)&bias_multiplier_, top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3] * sizeof(float)));
    CUDA_ERROR(cudaMemcpyAsync(bias_multiplier_, tmp_bias_multiplier.get(), top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3] * sizeof(float), cudaMemcpyHostToDevice, pNetResourceGpu->main_stream));

	cudaStreamSynchronize(pNetResourceGpu->main_stream);
    return CUDA_RETURN_VALUE;
}
int HolidayDeconvolutionGPU::Exit()
{
    if (ppfBlas) delete[]ppfBlas;
    if (ppfBlas_d) cudaFree(ppfBlas_d);
    if (ppfWeight_d) cudaFree(ppfWeight_d);
    if (pfWeight_d) cudaFree(pfWeight_d);
    if (pfBias_d) cudaFree(pfBias_d);
    if (bias_multiplier_) cudaFree(bias_multiplier_);
    
    return CUDA_RETURN_VALUE;
}

int HolidayDeconvolutionGPU::Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map)
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
    output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];
    output_data_map[0]->data_shape[1] = dwKernelSlices;
    output_data_map[0]->data_shape[2] = (input_data_map[0]->data_shape[2] - 1) * dwStrideH - dwPadH * 2 + (dwDilationH * (dwKernelRows - 1) + 1);
    output_data_map[0]->data_shape[3] = (input_data_map[0]->data_shape[3] - 1) * dwStrideW - dwPadW * 2 + (dwDilationW * (dwKernelCols - 1) + 1);
    output_data_map[0]->m_gpu.data_size = output_data_map[0]->data_shape[0] * output_data_map[0]->data_shape[1] * output_data_map[0]->data_shape[2] * output_data_map[0]->data_shape[3];
    gTmpBuffer_gpu(pNetResourceGpu, input_data_map[0]->data_shape[3] * input_data_map[0]->data_shape[2] * dwKernelCols*dwKernelRows*dwKernelSlices * dwGroup * pNetResourceGpu->dwMaxBatchNum * sizeof(float));
    float *pfDataTrans_d = (float *)pNetResourceGpu->pubyConvTmpBuffer;

    //gBiasSet_kernel << <CUDA_BLOCK(input_data_map[0]->data_shape[3] * input_data_map[0]->data_shape[2] * dwKernelCols*dwKernelRows*dwKernelSlices  * dwGroup * input_data_map[0]->data_shape[0], CUDA_THREAD_NUM),
    //    CUDA_THREAD_NUM, 0, pNetResourceGpu->main_stream>> >(pfBias_d, pfDataTrans_d,
    //        input_data_map[0]->data_shape[3] * input_data_map[0]->data_shape[2], 
    //        input_data_map[0]->data_shape[3] * input_data_map[0]->data_shape[2] * dwKernelCols*dwKernelRows*dwKernelSlices  * dwGroup,
    //        input_data_map[0]->data_shape[3] * input_data_map[0]->data_shape[2] * dwKernelCols*dwKernelRows*dwKernelSlices  * dwGroup * input_data_map[0]->data_shape[0]);

#ifdef _X64_

    int N = input_data_map[0]->data_shape[2] * input_data_map[0]->data_shape[3];
    int M = dwKernelSlices* dwKernelRows* dwKernelCols;
    int K = input_data_map[0]->data_shape[1]/dwGroup;

    //if (input_data_map[0]->data_shape[3] * input_data_map[0]->data_shape[2] * dwWeightRows / dwGroup < 10000)
    //{
    //    for (int i = 0; i < dwGroup * input_data_map[0]->data_shape[0]; i++)
    //    {
    //        ppfBlas[i] = (float *)input_data_map[0]->m_gpu.pfData_gpu + i * input_data_map[0]->data_shape[3] * input_data_map[0]->data_shape[2] * input_data_map[0]->data_shape[1] / dwGroup;
    //        ppfBlas[i + dwGroup *  input_data_map[0]->data_shape[0]] = pfDataTrans_d + i * input_data_map[0]->data_shape[3] * input_data_map[0]->data_shape[2] * dwWeightCols;
    //    }
    //    CUDA_ERROR(cudaMemcpy(ppfBlas_d, ppfBlas, dwGroup * input_data_map[0]->data_shape[0] * 2 * sizeof(float *), cudaMemcpyHostToDevice));
    //    float fAlpha = 1.f;
    //    float fBeta = 1.f;
    //    CUBLAS_ERROR(cublasSgemmBatched(pNetResourceGpu->Handle_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
    //        input_data_map[0]->data_shape[3] * input_data_map[0]->data_shape[2], dwKernelCols*dwKernelRows*dwKernelSlices, input_data_map[0]->data_shape[1] / dwGroup, &fAlpha,
    //        (const float **)ppfBlas_d, input_data_map[0]->data_shape[3] * input_data_map[0]->data_shape[2],
    //        (const float **)ppfWeight_d, dwKernelCols*dwKernelRows*dwKernelSlices, &fBeta,
    //        ppfBlas_d + dwGroup * input_data_map[0]->data_shape[0], 
    //        input_data_map[0]->data_shape[3] * input_data_map[0]->data_shape[2],
    //        dwGroup * input_data_map[0]->data_shape[0]));
    //}
    //else
    //{
    //    for (int i = 0; i < dwGroup * input_data_map[0]->data_shape[0]; i++)
    //    {
    //        float fAlpha = 1.f;
    //        float fBeta = 1.f;
    //        CUBLAS_ERROR(cublasSgemm(pNetResourceGpu->Handle_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
    //            N,
    //            M,
    //            K, 
    //            &fAlpha,
    //            (float *)input_data_map[0]->m_gpu.pfData_gpu + i * input_data_map[0]->data_shape[3] * input_data_map[0]->data_shape[2] * input_data_map[0]->data_shape[1] / dwGroup,
    //            N,
    //            pfWeight_d + (i % dwGroup) * input_data_map[0]->data_shape[1] * dwKernelSlices* dwKernelRows* dwKernelCols / dwGroup ,
    //            M, 
    //            &fBeta,
    //            pfDataTrans_d + i * input_data_map[0]->data_shape[3] * input_data_map[0]->data_shape[2] * dwKernelSlices* dwKernelRows* dwKernelCols,
    //            N));
    //    }
    //}

    for (int n = 0; n < input_data_map[0]->data_shape[0]; ++n)
    {

        float* col_buff = (float *)pNetResourceGpu->pubyConvTmpBuffer;

        int output_offset_ = input_data_map[0]->data_shape[1] * input_data_map[0]->data_shape[2] * input_data_map[0]->data_shape[3] / dwGroup;
        int weight_offset_ = dwKernelNum * dwKernelRows* dwKernelCols*dwKernelSlices / dwGroup;
        int col_offset_ = dwKernelSlices*dwKernelRows*dwKernelCols*input_data_map[0]->data_shape[2] * input_data_map[0]->data_shape[3];

        float* output = input_data_map[0]->m_gpu.pfData_gpu + n * input_data_map[0]->data_shape[1] * input_data_map[0]->data_shape[2] * input_data_map[0]->data_shape[3];

        for (int g = 0; g < dwGroup; ++g) {
            template_gpu_gemm<float>(pNetResourceGpu->Handle_cublas, true, false, M, N, K,
                (float)1., pfWeight_d + weight_offset_ * g, output + output_offset_* g,
                (float)0., col_buff + col_offset_ * g);
        }

        int top_dim = output_data_map[0]->data_shape[1] * output_data_map[0]->data_shape[2] * output_data_map[0]->data_shape[3];

        float* data = output_data_map[0]->m_gpu.pfData_gpu;

        //conv_col2im_gpu(col_buff, input);
        col2im_gpu(col_buff, dwKernelSlices,
            output_data_map[0]->data_shape[2], output_data_map[0]->data_shape[3],
            dwKernelRows,dwKernelCols,
            dwPadH, dwPadW,
            dwStrideH, dwStrideW,
            dwDilationH, dwDilationW, data+n*top_dim, pNetResourceGpu->main_stream);

        int single_image_size = output_data_map[0]->data_shape[2]* output_data_map[0]->data_shape[3];
       const float* bias = pfBias_d;
       //forward_gpu_bias(data + n * top_dim, bias);

       template_gpu_gemm<float>(pNetResourceGpu->Handle_cublas, false, false, dwKernelSlices, single_image_size
           , 1, (float)1., bias, bias_multiplier_,
           (float)1., data + n*top_dim);

    }
    

#else

    int N = input_data_map[0]->data_shape[2] * input_data_map[0]->data_shape[3];
    int M = dwKernelSlices* dwKernelRows* dwKernelCols;
    int K = input_data_map[0]->data_shape[1]/dwGroup;

	for (int n = 0; n < input_data_map[0]->data_shape[0]; ++n)
    {

        float* col_buff = (float *)pNetResourceGpu->pubyConvTmpBuffer;

        int output_offset_ = input_data_map[0]->data_shape[1] * input_data_map[0]->data_shape[2] * input_data_map[0]->data_shape[3] / dwGroup;
        int weight_offset_ = dwKernelNum * dwKernelRows* dwKernelCols*dwKernelSlices / dwGroup;
        int col_offset_ = dwKernelSlices*dwKernelRows*dwKernelCols*input_data_map[0]->data_shape[2] * input_data_map[0]->data_shape[3];

        float* output = input_data_map[0]->m_gpu.pfData_gpu + n * input_data_map[0]->data_shape[1] * input_data_map[0]->data_shape[2] * input_data_map[0]->data_shape[3];

		dim3 blocksize(CUDA_BLOCK(N, TRANS_BLOCK_DIM), CUDA_BLOCK(M, TRANS_BLOCK_DIM));
		dim3 threadsize(TRANS_BLOCK_DIM, TRANS_BLOCK_DIM);

        for (int g = 0; g < dwGroup; ++g) {
            // template_gpu_gemm<float>(pNetResourceGpu->Handle_cublas, true, false, M, N, K,
            //     (float)1., pfWeight_d + weight_offset_ * g, output + output_offset_* g,
            //     (float)0., col_buff + col_offset_ * g);
			gInnerProduct_TF_kernel << <blocksize, threadsize, 0, pNetResourceGpu->main_stream>> >(
				pfWeight_d + weight_offset_ * g,
				output + output_offset_* g,
				col_buff + col_offset_ * g,
				M, N, K, 0);
        }

        int top_dim = output_data_map[0]->data_shape[1] * output_data_map[0]->data_shape[2] * output_data_map[0]->data_shape[3];

        float* data = output_data_map[0]->m_gpu.pfData_gpu;

        //conv_col2im_gpu(col_buff, input);
        col2im_gpu(col_buff, dwKernelSlices,
            output_data_map[0]->data_shape[2], output_data_map[0]->data_shape[3],
            dwKernelRows,dwKernelCols,
            dwPadH, dwPadW,
            dwStrideH, dwStrideW,
            dwDilationH, dwDilationW, data+n*top_dim, pNetResourceGpu->main_stream);

        int single_image_size = output_data_map[0]->data_shape[2]* output_data_map[0]->data_shape[3];
       const float* bias = pfBias_d;
       //forward_gpu_bias(data + n * top_dim, bias);

	   dim3 bias_blocksize(CUDA_BLOCK(single_image_size, TRANS_BLOCK_DIM), CUDA_BLOCK(dwKernelSlices, TRANS_BLOCK_DIM));
	   dim3 bias_threadsize(TRANS_BLOCK_DIM, TRANS_BLOCK_DIM);

       //template_gpu_gemm<float>(pNetResourceGpu->Handle_cublas, false, false, dwKernelSlices, single_image_size
       //    , 1, (float)1., bias, bias_multiplier_,
       //    (float)1., data + n*top_dim);
	   gInnerProduct_FF_kernel << <bias_blocksize, bias_threadsize, 0, pNetResourceGpu->main_stream>> >(
		   bias,
		   bias_multiplier_,
		   data + n*top_dim,
		   dwKernelSlices, single_image_size, 1, 1);
    }

#endif
    //int dwSize = this->top_data_size[0].data_dim[1] * this->top_data_size[0].data_dim[2] * this->top_data_size[0].data_dim[3];
    //gOutputTrans_kernel << <CUDA_BLOCK(dwSize *  input_data_map[0]->data_shape[0], CUDA_THREAD_NUM), CUDA_THREAD_NUM, 0, pNetResourceGpu->main_stream>> >
    //    (pfDataTrans_d, (float *)output_data_map[0]->m_gpu.pfData_gpu, output_data_map[0]->m_gpu.data_size,
    //        input_data_map[0]->data_shape[2], input_data_map[0]->data_shape[3], dwKernelCols*dwKernelRows*dwKernelSlices  * dwGroup,
    //        output_data_map[0]->data_shape[2], output_data_map[0]->data_shape[3], dwKernelCols*dwKernelRows*dwKernelSlices  * dwGroup,
    //        dwStrideH, dwStrideW, dwPadH, dwPadW, dwDilationH, dwDilationW, dwKernelRows, dwKernelCols);

#ifdef _DEBUG
    cudaEventRecord(stop1, NULL);
    cudaEventSynchronize(stop1);
    float msecTotal1 = 0.0f;
    cudaEventElapsedTime(&msecTotal1, start1, stop1);
    printf(" Deconvolution: %f ms \n", msecTotal1);
#endif
#ifdef _DEBUG
    float *pfDataOut = new float[output_data_map[0]->m_gpu.data_size];
    output_data_map[0]->m_gpu.Gpu_DataOut(pNetResourceGpu, DATA_CPU_WIDTH, pfDataOut);
    delete[] pfDataOut;
    cudaDeviceSynchronize();
    printf("Deconvolution:%s\n", cudaGetErrorString(cudaGetLastError()));
#endif
    return CUDA_RETURN_VALUE;
}
