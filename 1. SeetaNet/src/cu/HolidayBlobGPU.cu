#include "MacroHoliday.h"

#ifdef HOLIDAY_GPU

#include "HolidayBlobGpu.h"
#include "HolidayCommonCuda.h"
#include "HolidayFeatureMap.h"

HolidayBlobGpu::~HolidayBlobGpu()
{

}


HolidayBlobGpu::HolidayBlobGpu()
{
	memory_size = 0;
	data_size = 0;

	pbyData_cpu = 0;
	pfData_cpu = 0;
	pfData_gpu = 0;
}

template<class T>
__global__ static void gMultiMatrixTrans_kernel(T *pDataIn, float *pfDataOut, int dwWidth, int dwHeight, int dwNum, int dwPitchIn, int dwPitchOut)
{
	__shared__ float pfTmp[TRANS_BLOCK_DIM][TRANS_BLOCK_DIM + 1];	//TRANS_BLOCK_DIM + 1  for  bank conflict

	int dwIdxN = blockIdx.y / CUDA_BLOCK(dwHeight, TRANS_BLOCK_DIM);
	int dwBlockS = blockDim.x * blockIdx.x;
	int dwBlockRC = blockDim.y * (blockIdx.y % CUDA_BLOCK(dwHeight, TRANS_BLOCK_DIM));
	int dwIdxX = dwBlockS + threadIdx.x;
	int dwIdxY = dwBlockRC + threadIdx.y;
	if (dwIdxX < dwWidth && dwIdxY < dwHeight)
	{
		pfTmp[threadIdx.y][threadIdx.x] = pDataIn[dwIdxN * dwHeight * dwPitchIn + dwIdxY * dwPitchIn + dwIdxX];
	}
	__syncthreads();
	dwIdxX = dwBlockS + threadIdx.y;
	dwIdxY = dwBlockRC + threadIdx.x;
	if (dwIdxX < dwWidth && dwIdxY < dwHeight)
	{
		pfDataOut[dwIdxN * dwWidth * dwPitchOut + dwIdxX * dwPitchOut + dwIdxY] = pfTmp[threadIdx.x][threadIdx.y];
	}
}

__global__ static void gCharToFloat_kernel(unsigned char *pbyDataIn, float *pfDataOut, int dwSize)
{
	int dwIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (dwIdx < dwSize)
	{
		pfDataOut[dwIdx] = pbyDataIn[dwIdx];
	}
}

int HolidayBlobGpu::Gpu_init(void *pNetResourceGpu)
{
	memory_size = 1;
	for (int i = 0; i < this->shape_.size(); i++)
	{
		memory_size *= shape_[i];
	}
	HolidayNetResourceGpu *ptNetResourceGpu = (HolidayNetResourceGpu*)pNetResourceGpu;
	GlobleBufferMallocGpu(ptNetResourceGpu, memory_size* sizeof(float));
	if (!pfData_gpu)
	{
		CUDA_ERROR(SafeCudaMalloc((void **)&pfData_gpu, memory_size * sizeof(float)));
		data_size = 0;
	}
	return CUDA_RETURN_VALUE;
}


int HolidayBlobGpu::Gpu_DataIn(void *pNetResourceGpu, int dwStorageType,void *pDataIn)
{
	int n_pitch;
	int n_slice;
	int n_height;
	int n_width;
	int n_num;
	if (shape_.size()==4)
	{
		n_pitch = shape_[1];
		n_slice = shape_[1];
		n_height= shape_[2];
		n_width = shape_[3];
		n_num	= shape_[0];
	}
	else if (shape_.size()==2)
	{
		n_pitch = shape_[1];
		n_slice = shape_[1];
		n_height = 1;
		n_width = 1;
		n_num = shape_[0];
	}
	else
	{

	}
	data_size = 1;
	for (int i = 0; i < this->shape_.size(); i++)
	{
		data_size *= shape_[i];
	}
	HolidayNetResourceGpu *ptNetResourceGpu = (HolidayNetResourceGpu *)pNetResourceGpu;
	switch (dwStorageType)
	{
		case DATA_CPU_WIDTH:
		{	
			CUDA_ERROR(cudaMemcpyAsync(pfData_gpu, pDataIn, data_size * sizeof(float), cudaMemcpyHostToDevice, ptNetResourceGpu->main_stream));
		    cudaStreamSynchronize(ptNetResourceGpu->main_stream);
			break;
		}
		case DATA_CPU_WIDTH_CHAR:
		{
			unsigned char *pubyTmp = (unsigned char *)ptNetResourceGpu->pubyConvTmpBuffer;
			CUDA_ERROR(cudaMemcpyAsync(pubyTmp, pbyData_cpu, data_size * sizeof(unsigned char), cudaMemcpyHostToDevice, ptNetResourceGpu->main_stream));
			gCharToFloat_kernel << <CUDA_BLOCK(data_size, CUDA_THREAD_NUM), CUDA_THREAD_NUM, 0, ptNetResourceGpu->main_stream>> >(pubyTmp, pfData_gpu, data_size);
			break;
		}
		case DATA_CPU_SLICE_CHAR:
		{
			data_size = shape_[0] * shape_[1] * shape_[2] * shape_[3];
			gTmpBuffer_gpu(ptNetResourceGpu, data_size * sizeof(unsigned char));
			unsigned char *pubyTmp = (unsigned char*)ptNetResourceGpu->pubyConvTmpBuffer;
			CUDA_ERROR(cudaMemcpyAsync(pubyTmp, pDataIn, data_size * sizeof(unsigned char), cudaMemcpyHostToDevice, ptNetResourceGpu->main_stream));
			dim3 blocksize(CUDA_BLOCK(shape_[1], TRANS_BLOCK_DIM), shape_[0] * CUDA_BLOCK(shape_[3] * shape_[2], TRANS_BLOCK_DIM));
			dim3 threadsize(TRANS_BLOCK_DIM, TRANS_BLOCK_DIM);
			gMultiMatrixTrans_kernel<unsigned char> << <blocksize, threadsize, 0, ptNetResourceGpu->main_stream>> >(pubyTmp, (float *)pfData_gpu, shape_[1], shape_[3] * shape_[2], shape_[0], shape_[1], shape_[3] * shape_[2]);
			//unsigned char *pubyTmp = (unsigned char *)pNetResourceGpu->pubyConvTmpBuffer;
			//CUDA_ERROR(cudaMemcpy(pubyTmp, pbyData_cpu, data_size* sizeof(unsigned char), cudaMemcpyHostToDevice));
			//dim3 blocksize(CUDA_BLOCK(n_slice, TRANS_BLOCK_DIM), n_num * CUDA_BLOCK(n_height * n_width, TRANS_BLOCK_DIM));
			//dim3 threadsize(TRANS_BLOCK_DIM, TRANS_BLOCK_DIM);
			//gMultiMatrixTrans_kernel << <blocksize, threadsize, 0, pNetResourceGpu->main_stream>> >(pubyTmp, pfData_gpu, n_slice, n_height * n_width, n_num, n_pitch, n_height * n_width);
			break;
		}
		case DATA_GPU:
			break;
		case DATA_OPENCL:
			break;
		default:
			break;
	}
	return CUDA_RETURN_VALUE;
}
int HolidayBlobGpu::Gpu_free()
{
	if (pfData_gpu) cudaFree(pfData_gpu);
	pfData_gpu = NULL;
	return CUDA_RETURN_VALUE;
}
int HolidayBlobGpu::Gpu_DataOut(void *pNetResourceGpu_in, int dwStorageType, float *out)
{
	int n_pitch;
	int n_slice;
	int n_height;
	int n_width;
	int n_num;
	if (shape_.size() == 4)
	{
		n_pitch = shape_[1];
		n_slice = shape_[1];
		n_height = shape_[2];
		n_width = shape_[3];
		n_num = shape_[0];
	}
	else if (shape_.size() == 2)
	{
		n_pitch = shape_[1];
		n_slice = shape_[1];
		n_height = 1;
		n_width = 1;
		n_num = shape_[0];
	}
	else
	{

	}
	HolidayNetResourceGpu *pNetResourceGpu = (HolidayNetResourceGpu *)pNetResourceGpu_in;
	switch (dwStorageType)
	{
	case DATA_CPU_WIDTH:
	{
		CUDA_ERROR(cudaMemcpyAsync(out, pfData_gpu, n_slice * n_height * n_width * n_num * sizeof(float), cudaMemcpyDeviceToHost, pNetResourceGpu->main_stream));
		cudaStreamSynchronize(pNetResourceGpu->main_stream);
		break;
	}
	case DATA_CPU_SLICE:
	{
		float *pfTmp = (float *)pNetResourceGpu->pubyConvTmpBuffer;
		dim3 blocksize(CUDA_BLOCK(n_height * n_width, TRANS_BLOCK_DIM), n_num * CUDA_BLOCK(n_slice, TRANS_BLOCK_DIM));
		dim3 threadsize(TRANS_BLOCK_DIM, TRANS_BLOCK_DIM);
		gMultiMatrixTrans_kernel<float> << <blocksize, threadsize, 0, pNetResourceGpu->main_stream>> >(pfData_gpu, pfTmp, n_height * n_width, n_slice, n_num, n_height * n_width, n_pitch);
		CUDA_ERROR(cudaMemcpyAsync(out, pfTmp, n_pitch * n_height * n_width * n_num * sizeof(float), cudaMemcpyDeviceToHost, pNetResourceGpu->main_stream));
		cudaStreamSynchronize(pNetResourceGpu->main_stream);
		break;
	}
	case DATA_GPU:
		break;
	case DATA_OPENCL:
		break;
	default:
		break;
	}
	return CUDA_RETURN_VALUE;
}

#endif // HOLIDAY_GPU
