#include"HolidayBatchToSpaceNDGPU.h"
#include"HolidayCommonCuda.h"

HolidayBatchToSpaceNDGPU::HolidayBatchToSpaceNDGPU()
{

}

HolidayBatchToSpaceNDGPU::~HolidayBatchToSpaceNDGPU()
{

}

void HolidayBatchToSpaceNDGPU::CaculateOutputSize(int input_number, int input_height, int input_width, int input_channels,
													 int &output_number, int &output_height, int &output_width, int &output_channels)
{
    output_number = input_number / (m_block_shape[0] * m_block_shape[1]);
    output_height = input_height * m_block_shape[0] - m_crops[0] - m_crops[1];
    output_width = input_width * m_block_shape[1] - m_crops[2] - m_crops[3];
    output_channels = input_channels;
}

int HolidayBatchToSpaceNDGPU::Init(Holiday_LayerParameter& inputparam, HolidayNetResource<float> *pNetResource)
{
	pNetResourceGpu = (HolidayNetResourceGpu *)pNetResource->pNetResourceGpu;

    // set bottom size
    int bottom_index = inputparam.bottom_index(0);
    HolidayDataSize bottom_size = pNetResource->feature_vector_size[bottom_index];
    this->bottom_data_size.resize(1);
    this->bottom_data_size[0] = bottom_size;

    // read param from inputparam.spacetobatchnd_param();
    auto &param = inputparam.batchtospacend_param();
    for (int i = 0; i < param.block_shape_size(); i++)
    {
        m_block_shape.push_back(param.block_shape(i));
    }
    for (int i = 0; i < param.crops_size(); i++)
    {
        m_crops.push_back(param.crops(i));
    }

    assert(m_block_shape.size() == 2 && m_crops.size() == 4);
    assert(m_crops[0] >= 0 && m_crops[1] >= 0 && m_crops[2] >= 0 && m_crops[3] >= 0);

	CUDA_ERROR(SafeCudaMalloc((void **)&m_block_shape_gpu, m_block_shape.size() * sizeof(int)));
	CUDA_ERROR(SafeCudaMalloc((void **)&m_crops_gpu, m_crops.size() * sizeof(int)));
	CUDA_ERROR(cudaMemcpyAsync(m_block_shape_gpu, m_block_shape.data(), m_block_shape.size() * sizeof(int), cudaMemcpyHostToDevice, pNetResourceGpu->main_stream));
	CUDA_ERROR(cudaMemcpyAsync(m_crops_gpu, m_crops.data(), m_crops.size() * sizeof(int), cudaMemcpyHostToDevice, pNetResourceGpu->main_stream));

    // set top size
    this->top_data_size.resize(1);
    this->top_data_size[0].data_dim.resize(4);
    CaculateOutputSize(this->bottom_data_size[0].data_dim, this->top_data_size[0].data_dim);

	
    cudaStreamSynchronize(pNetResourceGpu->main_stream);
    return CUDA_RETURN_VALUE;
}

int HolidayBatchToSpaceNDGPU::Exit() {
	if (m_block_shape_gpu) cudaFree(m_block_shape_gpu);
	if (m_crops_gpu) cudaFree(m_crops_gpu);

	return CUDA_RETURN_VALUE;
}

// number, channels, height, width means
__global__ static void gBatchToSpaceND_kernel(float *input_data, float *output_data,
											  int input_number, int input_channels, int input_height, int input_width,
											  int output_number, int output_channels, int output_height, int output_width,
											  int input_size, int input_number_step, int input_channels_step, int input_height_step,
											  int output_size, int output_number_step, int output_channels_step, int output_height_step,
											  int *block_shape, int *crops)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	auto &B = block_shape;
	auto &C = crops;

	if (index < output_size)
	{
		int at_output_i = index;

		int n = index / output_number_step;
		index %= output_number_step;
		int c = index / output_channels_step;
		index %= output_channels_step;
		int h = index / output_height_step;
		index %= output_height_step;
		int w = index;

        int in = ((h + C[0]) % B[0] * B[1] + (w + C[2]) % B[1]) * output_number + n;
        int ic = c;
        int ih = (h + C[0]) / B[0];
        int iw = (w + C[2]) / B[1];

        int at_input_i = in * input_number_step
                         + ic * input_channels_step
                         + ih * input_height_step
                         + iw;

        output_data[at_output_i] = input_data[at_input_i];
	}
}

int HolidayBatchToSpaceNDGPU::Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map)
{
	// trans param to gpu
	input_data_map[0]->m_gpu.shape_ = input_data_map[0]->data_shape;
	input_data_map[0]->m_gpu.Gpu_DataIn(pNetResourceGpu, input_data_map[0]->dwStorageType, input_data_map[0]->m_cpu.dataMemoryPtr());
	input_data_map[0]->dwStorageType = DATA_GPU;
	output_data_map[0]->dwStorageType = DATA_GPU;

	// set output data type and shape

	// set output data shape
	CaculateOutputSize(input_data_map[0]->data_shape, output_data_map[0]->data_shape);

	// set output gpu shape
	output_data_map[0]->m_gpu.shape_ = output_data_map[0]->data_shape;
	output_data_map[0]->m_gpu.data_size = output_data_map[0]->data_shape[0] * output_data_map[0]->data_shape[1] * output_data_map[0]->data_shape[2] * output_data_map[0]->data_shape[3];


	// write output
	int input_number = input_data_map[0]->data_shape[0];
	int input_channels = input_data_map[0]->data_shape[1];
	int input_height = input_data_map[0]->data_shape[2];
	int input_width = input_data_map[0]->data_shape[3];

	int output_number = output_data_map[0]->data_shape[0];
	int output_channels = output_data_map[0]->data_shape[1];
	int output_height = output_data_map[0]->data_shape[2];
	int output_width = output_data_map[0]->data_shape[3];

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

	gBatchToSpaceND_kernel << <CUDA_BLOCK(output_size, CUDA_THREAD_NUM), CUDA_THREAD_NUM, 0, pNetResourceGpu->main_stream>> >
			(input_data, output_data,
					input_number, input_channels, input_height, input_width,
					output_number, output_channels, output_height, output_width,
					input_size, input_number_step, input_channels_step, input_height_step,
					output_size, output_number_step, output_channels_step, output_height_step,
					m_block_shape_gpu, m_crops_gpu);

	return CUDA_RETURN_VALUE;
}
