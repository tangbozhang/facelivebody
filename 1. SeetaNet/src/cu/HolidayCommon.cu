#include "HolidayCommonCuda.h"

int GlobleBufferMallocGpu(HolidayNetResourceGpu *pNetResourceGpu, int dwRequiredSize)
{
	if (pNetResourceGpu->dwBufferSize < dwRequiredSize)
	{
		if (pNetResourceGpu->pubyConvTmpBuffer) cudaFree(pNetResourceGpu->pubyConvTmpBuffer);
		pNetResourceGpu->dwBufferSize = dwRequiredSize;
		CUDA_ERROR(SafeCudaMalloc((void **)&pNetResourceGpu->pubyConvTmpBuffer, pNetResourceGpu->dwBufferSize));
	}
	return CUDA_RETURN_VALUE;
}

int gTmpBuffer_gpu(HolidayNetResourceGpu *pNetResourceGpu, int dwRequiredSize)
{
	if (pNetResourceGpu->dwBufferSize < dwRequiredSize)
	{
		if (pNetResourceGpu->pubyConvTmpBuffer) cudaFree(pNetResourceGpu->pubyConvTmpBuffer);
		pNetResourceGpu->dwBufferSize = dwRequiredSize;
		CUDA_ERROR(SafeCudaMalloc((void **)&pNetResourceGpu->pubyConvTmpBuffer, pNetResourceGpu->dwBufferSize));
	}
	return CUDA_RETURN_VALUE;
}

#define MAX_STREAM_NUM	50
int HolidayNetGPUInit(Holiday_NetParam_gpu *pdjNetParam, void **ppNetResource)
{
	HolidayNetResourceGpu *pNetResource;
	pNetResource = new HolidayNetResourceGpu;
	pNetResource->dwMaxBatchNum = pdjNetParam->dwMaxBatchNum;
#ifdef _DEBUG
	CUDA_ERROR(cudaStreamCreateWithFlags(&pNetResource->main_stream, cudaStreamDefault));
#else
	// CUDA_ERROR(cudaStreamCreateWithFlags(&pNetResource->main_stream, cudaStreamDefault));
	CUDA_ERROR(cudaStreamCreateWithFlags(&pNetResource->main_stream, cudaStreamNonBlocking));
#endif
#ifdef _X64_
	CUBLAS_ERROR(cublasCreate(&(pNetResource->Handle_cublas)));
	cublasSetStream(pNetResource->Handle_cublas, pNetResource->main_stream);
#endif
#ifdef GPU_CUDNN
	CUDNN_ERROR(cudnnCreate(&(pNetResource->Handle_cudnn)));
#endif

	pNetResource->dwBufferSize = 0;
	pNetResource->pubyConvTmpBuffer = 0;

	*ppNetResource = pNetResource;

	return CUDA_RETURN_VALUE;
}
/**
 * Synchronize GPU stream
 */
void HolidayNetGPUSync(Holiday_NetResource_gpu *gpu)
{
	cudaStreamSynchronize(gpu->main_stream);
}

/************************************************************************/
/* Exit NET                                                            */
/************************************************************************/
int HolidayNetGPUExit(void *pNetResource)
{
	Holiday_NetResource_gpu *ptmpNetResource = (Holiday_NetResource_gpu *)pNetResource;
#ifdef _X64_
	CUBLAS_ERROR(cublasDestroy(ptmpNetResource->Handle_cublas));
#endif
#ifdef GPU_CUDNN
	CUDNN_ERROR(cudnnDestroy(pNetResource->Handle_cudnn));
#endif
	if (ptmpNetResource->pubyConvTmpBuffer) cudaFree(ptmpNetResource->pubyConvTmpBuffer);
	cudaStreamDestroy(ptmpNetResource->main_stream);
	delete ptmpNetResource;

	return CUDA_RETURN_VALUE;
}

#ifdef _X64_

template <>
int template_gpu_gemm<float>(cublasHandle_t handle, bool TransA,
    bool TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
    // Note that cublas follows fortran order.
    int lda = (TransA == false) ? K : M;
    int ldb = (TransB == false) ? N : K;
    cublasOperation_t cuTransA =
        (TransA == false) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuTransB =
        (TransB == false) ? CUBLAS_OP_N : CUBLAS_OP_T;
    CUBLAS_ERROR(cublasSgemm(handle, cuTransB, cuTransA,
        N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));

    return 0;
}

template <>
int template_gpu_gemm<double>(cublasHandle_t handle, bool TransA,
    bool TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
    // Note that cublas follows fortran order.
    int lda = (TransA == false) ? K : M;
    int ldb = (TransB == false) ? N : K;
    cublasOperation_t cuTransA =
        (TransA == false) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuTransB =
        (TransB == false) ? CUBLAS_OP_N : CUBLAS_OP_T;
    CUBLAS_ERROR(cublasDgemm(handle, cuTransB, cuTransA,
        N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
    return 0;
}

#endif

template <typename Dtype>
__global__ void col2im_gpu_kernel(const int n, const Dtype* data_col,
    const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    Dtype* data_im)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
        index < (n);
        index += blockDim.x * gridDim.x)
    {
        Dtype val = 0;
        const int w_im = index % width + pad_w;
        const int h_im = (index / width) % height + pad_h;
        const int c_im = index / (width * height);
        int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
        int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
        // compute the start and end of the output
        const int w_col_start =
            (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
        const int w_col_end = min(w_im / stride_w + 1, width_col);
        const int h_col_start =
            (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
        const int h_col_end = min(h_im / stride_h + 1, height_col);
        // TODO: use LCM of stride and dilation to avoid unnecessary loops
        for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
            for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
                int h_k = (h_im - h_col * stride_h);
                int w_k = (w_im - w_col * stride_w);
                if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
                    h_k /= dilation_h;
                    w_k /= dilation_w;
                    int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
                        height_col + h_col) * width_col + w_col;
                    val += data_col[data_col_index];
                }
            }
        }
        data_im[index] = val;
    }
}

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    Dtype* data_im, cudaStream_t stream) {
    int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) /
        stride_h + 1;
    int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) /
        stride_w + 1;
    int num_kernels = channels * height * width;
    // To avoid involving atomic operations, we will launch one kernel per
    // bottom dimension, and then in the kernel add up the top dimensions.
    // NOLINT_NEXT_LINE(whitespace/operators)
    col2im_gpu_kernel<Dtype> << <CUDA_BLOCK(num_kernels, CUDA_THREAD_NUM),
        CUDA_THREAD_NUM, 0, stream>> >(
            num_kernels, data_col, height, width, channels, kernel_h, kernel_w,
            pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
            height_col, width_col, data_im);
    
}

// Explicit instantiation
template void col2im_gpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_im, cudaStream_t stream);
template void col2im_gpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    double* data_im, cudaStream_t stream);