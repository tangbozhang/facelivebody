#include "HolidayRealMulGPU.h"
#include <cuda_runtime_api.h>

#include <cfloat>
#include <climits>

HolidayRealMulGPU::HolidayRealMulGPU()
{
	pf_y_gpu = nullptr;
}
HolidayRealMulGPU::~HolidayRealMulGPU()
{
}

__global__ static void gRealMul_single_kernel(float *pfDataIn, float *pfDataOut, int dwSize,
	float y)
{
	int dwIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (dwIdx < dwSize)
	{
        pfDataOut[dwIdx] = pfDataIn[dwIdx] * y;
	}
}

__global__ static void gRealMul_broadcast_kernel(float *input_data, float *output_data,
    int input_number, int input_channels, int input_height, int input_width,
    int input_size, int input_number_step, int input_channels_step, int input_height_step,
    float *y,
    int y_number, int y_channels, int y_height, int y_width,
    int y_size, int y_number_step, int y_channels_step, int y_height_step)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < input_size)
    {
        int at_input_i = index;

        int n = index / input_number_step;
        index %= input_number_step;
        int c = index / input_channels_step;
        index %= input_channels_step;
        int h = index / input_height_step;
        index %= input_height_step;
        int w = index;

        auto yn = n % y_number;
        auto yc = c % y_channels;
        auto yh = h % y_height;
        auto yw = w % y_width;

        int y_index = yn * y_number_step + yc * y_channels_step + yh * y_height_step + yw;

        output_data[at_input_i] = input_data[at_input_i] * y[y_index];
    }
}

int HolidayRealMulGPU::Init(Holiday_LayerParameter &inputparam, HolidayNetResource<float> *pNetResource)
{
    auto &blob_y = inputparam.real_mul_param().y();
    auto &blob_y_shape = blob_y.shape();
    m_y_shape.resize(blob_y_shape.dim_size());
    for (size_t i = 0; i < m_y_shape.size(); ++i)
    {
        m_y_shape[i] = blob_y_shape.dim(i);
    }
    assert(m_y_shape.size() <= 4);
    while (m_y_shape.size() < 4)
    {
        m_y_shape.push_back(1);
    }

    int length_y = blob_y.data_size();
    m_y_data.reset(new float[length_y], std::default_delete<float[]>());
    for (int i = 0; i < length_y; i++)
    {
        auto tmp_y_value = blob_y.data(i);
        if (tmp_y_value < FLT_EPSILON && -tmp_y_value < FLT_EPSILON) tmp_y_value = 0;
        m_y_data.get()[i] = tmp_y_value;
    }

	pNetResourceGpu = (HolidayNetResourceGpu *)pNetResource->pNetResourceGpu;
	
	int index = inputparam.bottom_index(0);
	bottom_data_size.resize(1);
	bottom_data_size[0] = pNetResource->feature_vector_size[index];

    auto y_count = m_y_shape[0] * m_y_shape[1] * m_y_shape[2] * m_y_shape[3];

	top_data_size = bottom_data_size;
    CUDA_ERROR(SafeCudaMalloc((void **)&pf_y_gpu, y_count * sizeof(float)));
    CUDA_ERROR(cudaMemcpyAsync(pf_y_gpu, m_y_data.get(), y_count * sizeof(float), cudaMemcpyHostToDevice, pNetResourceGpu->main_stream));

	cudaStreamSynchronize(pNetResourceGpu->main_stream);
	return CUDA_RETURN_VALUE;
}
int HolidayRealMulGPU::Exit()
{
    if (pf_y_gpu) cudaFree(pf_y_gpu);
    pf_y_gpu = nullptr;

	return CUDA_RETURN_VALUE;
}

int HolidayRealMulGPU::Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map)
{
#ifdef _DEBUG
	cudaEvent_t start1;
	cudaEventCreate(&start1);
	cudaEvent_t stop1;
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, NULL);
#endif
	int dwsize1 = top_data_size[0].data_dim[1] * top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3];

	input_data_map[0]->m_gpu.shape_ = input_data_map[0]->data_shape;
	input_data_map[0]->m_gpu.Gpu_DataIn(pNetResourceGpu, input_data_map[0]->dwStorageType, input_data_map[0]->m_cpu.dataMemoryPtr());
	input_data_map[0]->dwStorageType = DATA_GPU;
	output_data_map[0]->dwStorageType = DATA_GPU;
	output_data_map[0]->data_shape = input_data_map[0]->data_shape;
	output_data_map[0]->m_gpu.shape_ = output_data_map[0]->data_shape;
	output_data_map[0]->m_gpu.data_size = input_data_map[0]->m_gpu.data_size;

    int count_y = m_y_shape[0] * m_y_shape[1] * m_y_shape[2] * m_y_shape[3];

    if (count_y == 1)
    {
        gRealMul_single_kernel << <CUDA_BLOCK(output_data_map[0]->m_gpu.data_size, CUDA_THREAD_NUM), CUDA_THREAD_NUM, 0, pNetResourceGpu->main_stream>> >
            ((float *)input_data_map[0]->m_gpu.pfData_gpu, (float *)output_data_map[0]->m_gpu.pfData_gpu,
            output_data_map[0]->m_gpu.data_size, *m_y_data.get());
    }
    else
    {
        // write output
        int input_number = input_data_map[0]->data_shape[0];
        int input_channels = input_data_map[0]->data_shape[1];
        int input_height = input_data_map[0]->data_shape[2];
        int input_width = input_data_map[0]->data_shape[3];

        int y_number = m_y_shape[0];
        int y_channels = m_y_shape[1];
        int y_height = m_y_shape[2];
        int y_width = m_y_shape[3];

        int input_size = input_number * input_channels * input_height * input_width;
        int input_number_step = input_channels * input_height * input_width;
        int input_channels_step = input_height * input_width;
        int input_height_step = input_width;

        int y_size = y_number * y_channels * y_height * y_width;
        int y_number_step = y_channels * y_height * y_width;
        int y_channels_step = y_height * y_width;
        int y_height_step = y_width;

        float *input_data = input_data_map[0]->m_gpu.pfData_gpu;
        float *output_data = output_data_map[0]->m_gpu.pfData_gpu;
        gRealMul_broadcast_kernel << <CUDA_BLOCK(output_data_map[0]->m_gpu.data_size, CUDA_THREAD_NUM), CUDA_THREAD_NUM, 0, pNetResourceGpu->main_stream>> >
            (input_data, output_data,
            input_number, input_channels, input_height, input_width,
            input_size, input_number_step, input_channels_step, input_height_step,
            pf_y_gpu,
            y_number, y_channels, y_height, y_width,
            y_size, y_number_step, y_channels_step, y_height_step);
    }

#ifdef _DEBUG
	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
	float msecTotal1 = 0.0f;
	cudaEventElapsedTime(&msecTotal1, start1, stop1);
	printf(" RealMul: %f ms \n", msecTotal1);
#endif
#ifdef _DEBUG
	int dwsize = top_data_size[0].data_dim[1] * top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3];
	float *pfDataOut = new float[dwsize];
	cudaMemcpy(pfDataOut, output_data_map[0]->m_gpu.pfData_gpu,dwsize * sizeof(float), cudaMemcpyDeviceToHost);
	delete[] pfDataOut;
	cudaDeviceSynchronize();
	printf("RealMul:%s\n", cudaGetErrorString(cudaGetLastError()));
#endif
	return CUDA_RETURN_VALUE;
}
