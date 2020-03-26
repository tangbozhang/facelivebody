#ifndef HOLIDAY_COMMONCUDA_CUH
#define HOLIDAY_COMMONCUDA_CUH

#include "cuda_runtime.h"					// make_cudaExtent make_cudaPitchedPtr
#include "cuda_runtime_api.h"
#ifdef GPU_CUDNN
#include "cudnn.h"
#pragma comment(lib,"cudnn.lib")
#endif

#include "thrust/functional.h"
#include "thrust/sort.h"
#include "device_launch_parameters.h"		// threadidx.x 
#include "texture_indirect_functions.h"		// tex3D
//#include "sm_32_atomic_functions.hpp"		// atomicadd 1.1+	__shfl_down 3.0+
//#include "sm_32_intrinsics.hpp"				// __ldg
#ifdef _X64_
#include "cublas_v2.h"						// cublasSgemv
#pragma comment(lib,"cublas.lib")
#endif
#include "device_functions.h"				// __syncthreads
#pragma comment(lib,"cuda.lib")
#pragma comment(lib,"cudart.lib")

#include "except.h"


typedef struct HolidayNetResourceGpu
{
	int dwMaxBatchNum;
	// int dwStreamNum;
	// cudaStream_t *pStream;
    cudaStream_t main_stream;
#ifdef _X64_
	cublasHandle_t Handle_cublas;
#endif
#ifdef GPU_CUDNN
	cudnnHandle_t Handle_cudnn;
#endif
	unsigned char *pubyConvTmpBuffer;
	int dwBufferSize;
}Holiday_NetResource_gpu;

typedef struct Holiday_NetParam_gpu
{
	int dwMaxBatchNum;
}Holiday_NetParam_gpu;

#define CUDA_BLOCK(a, b)	(((a) + (b) - 1) / (b))
#define CUDA_THREAD_NUM			(512)
#define TRANS_BLOCK_DIM			(16)
#define TENSOR_DIMS_CUDNN		(4)

#define CUDA_NOERR					0x00000000
#define CUDA_ERR						0x83ffffff
#define CUDA_ERR_SEGEMENT_1			0x83000000
#define CUDA_ERR_SEGEMENT_2			0x84000000
#define CUDA_ERR_SEGEMENT_3			0x85000000

#define CUDA_ERROR(a) {int b = a; if ((b) != 0) return (abs(b));}
#define CUBLAS_ERROR(a) {int b = a; if ((b) != 0) return (abs(b));}
#define CUDNN_ERROR(a) {int b = a; if ((b) != 0) return (abs(b));}
#define CUDA_RETURN_VALUE	(cudaPeekAtLastError() ? (cudaPeekAtLastError()) : CUDA_NOERR)

int GlobleBufferMallocGpu(HolidayNetResourceGpu *pNetResourceGpu, int dwRequiredSize);
int gTmpBuffer_gpu(HolidayNetResourceGpu *pNetResourceGpu, int dwRequiredSize);
#define Holliday_MAX(a, b)		((a) > (b) ? (a) : (b))
#define Holliday_MIN(a, b)		((a) < (b) ? (a) : (b))

int HolidayNetGPUInit(Holiday_NetParam_gpu *pdjNetParam, void **ppNetResource);

int HolidayNetGPUExit(void *pNetResource);

void HolidayNetGPUSync(Holiday_NetResource_gpu *gpu);


#ifdef _X64_
template <typename Dtype>
int template_gpu_gemm(cublasHandle_t handle, const bool TransA,
    const bool TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C);
#endif

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    Dtype* data_im, cudaStream_t stream);

inline cudaError_t SafeCudaMalloc(void **devPtr, size_t size)
{
    *devPtr = nullptr;
    auto err = cudaMalloc(devPtr, size);
    if (size != 0 && *devPtr == nullptr) throw orz::OutOfMemoryException(size, "gpu");
    return err;
}

#endif