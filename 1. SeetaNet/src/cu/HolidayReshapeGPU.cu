#include"HolidayReshapeGPU.h"
#include "HolidayCommonCuda.h"
#include <cuda_runtime_api.h>

HolidayReshapeGPU::HolidayReshapeGPU()
{
}
HolidayReshapeGPU::~HolidayReshapeGPU()
{
}

int HolidayReshapeGPU::Init(Holiday_LayerParameter& inputparam, HolidayNetResource<float> *pNetResource)
{
	pNetResourceGpu = (HolidayNetResourceGpu *)pNetResource->pNetResourceGpu;
	
	int index = inputparam.bottom_index(0);
	bottom_data_size.resize(1);
	bottom_data_size[0] = pNetResource->feature_vector_size[index];

    auto reshape_param = inputparam.reshape_param();
    m_shape.resize(reshape_param.shape_size());
    for (size_t i = 0; i < m_shape.size(); ++i)
    {
        m_shape[i] = reshape_param.shape(i);
    }
    assert(m_shape.size() == 4);
    assert(m_shape[0] == 1);

    m_permute.resize(reshape_param.permute_size());
    for (size_t i = 0; i < m_permute.size(); ++i)
    {
        m_permute[i] = reshape_param.permute(i);
    }
    assert(m_permute.empty() || m_permute.size() == 4);
    
	this->top_data_size.resize(1);
    this->top_data_size[0].data_dim.resize(4);
    this->top_data_size[0].data_dim[0] = bottom_data_size[0].data_dim[0];
    this->top_data_size[0].data_dim[1] = m_shape[1];
    this->top_data_size[0].data_dim[2] = m_shape[2];
    this->top_data_size[0].data_dim[3] = m_shape[3];

	return CUDA_RETURN_VALUE;
}
int HolidayReshapeGPU::Exit()
{
	return CUDA_RETURN_VALUE;
}

__global__ static void gPermute_kernel(float *input_data, float *output_data,
    int input_number, int input_channels, int input_height, int input_width,
    int output_number, int output_channels, int output_height, int output_width,
    int input_size, int input_number_step, int input_channels_step, int input_height_step,
    int output_size, int output_number_step, int output_channels_step, int output_height_step,
    int dim1, int dim2, int dim3, int dim4)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    int transdim[4] = { dim1, dim2, dim3, dim4 };   // 0, 2, 3, 1

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

        int encode_in[4] = { n, c, h, w }; 
        int out_index[4] = { encode_in[transdim[0]], encode_in[transdim[1]], encode_in[transdim[2]], encode_in[transdim[3]] };

        int at_output_i = out_index[0] * output_number_step + out_index[1] * output_channels_step + out_index[2] * output_height_step + out_index[3];

        output_data[at_output_i] = input_data[at_input_i];
    }
}

int HolidayReshapeGPU::Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map)
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


    if (!m_permute.empty())
    {
        // write output
        int input_number = input_data_map[0]->data_shape[0];
        int input_channels = input_data_map[0]->data_shape[1];
        int input_height = input_data_map[0]->data_shape[2];
        int input_width = input_data_map[0]->data_shape[3];

        int output_number = input_data_map[0]->data_shape[m_permute[0]];
        int output_channels = input_data_map[0]->data_shape[m_permute[1]];
        int output_height = input_data_map[0]->data_shape[m_permute[2]];
        int output_width = input_data_map[0]->data_shape[m_permute[3]];

        int input_size = input_number * input_channels * input_height * input_width;
        int input_number_step = input_channels * input_height * input_width;
        int input_channels_step = input_height * input_width;
        int input_height_step = input_width;

        int output_size = output_number * output_channels * output_height * output_width;
        int output_number_step = output_channels * output_height * output_width;
        int output_channels_step = output_height * output_width;
        int output_height_step = output_width;

        float *input_data = input_data_map[0]->m_gpu.pfData_gpu;
        float *output_data = output_data_map[0]->m_gpu.pfData_gpu;

        gPermute_kernel << <CUDA_BLOCK(input_size, CUDA_THREAD_NUM), CUDA_THREAD_NUM >> >
            (input_data, output_data,
            input_number, input_channels, input_height, input_width,
            output_number, output_channels, output_height, output_width,
            input_size, input_number_step, input_channels_step, input_height_step,
            output_size, output_number_step, output_channels_step, output_height_step,
            m_permute[0], m_permute[1], m_permute[2], m_permute[3]);
    }
    else
    {
        if (input_data_map[0]->m_gpu.pfData_gpu != output_data_map[0]->m_gpu.pfData_gpu)
        {
            cudaMemcpyAsync(output_data_map[0]->m_gpu.pfData_gpu, input_data_map[0]->m_gpu.pfData_gpu, output_data_map[0]->m_gpu.data_size * sizeof(float), cudaMemcpyDeviceToDevice, pNetResourceGpu->main_stream);
        }
    }

	output_data_map[0]->dwStorageType = DATA_GPU;

    output_data_map[0]->data_shape.resize(4);
    output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];
    output_data_map[0]->data_shape[1] = m_shape[1];
    output_data_map[0]->data_shape[2] = m_shape[2];
    output_data_map[0]->data_shape[3] = m_shape[3];

	output_data_map[0]->m_gpu.shape_ = output_data_map[0]->data_shape;
	output_data_map[0]->m_gpu.data_size = input_data_map[0]->m_gpu.data_size;

#ifdef _DEBUG
	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
	float msecTotal1 = 0.0f;
	cudaEventElapsedTime(&msecTotal1, start1, stop1);
	printf(" Reshape: %f ms \n", msecTotal1);
#endif
#ifdef _DEBUG
	int dwSize1 = top_data_size[0].data_dim[1] * top_data_size[0].data_dim[2] * top_data_size[0].data_dim[3];
	float *pfDataOut = new float[dwSize1];
	cudaMemcpy(pfDataOut, output_data_map[0]->m_gpu.pfData_gpu, dwSize1 * sizeof(float), cudaMemcpyDeviceToHost);
	delete[] pfDataOut;
	cudaDeviceSynchronize();
	printf("Reshape:%s\n", cudaGetErrorString(cudaGetLastError()));
#endif
	return CUDA_RETURN_VALUE;
}
