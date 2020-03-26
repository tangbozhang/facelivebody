#ifndef __HOLIDAY_BATCH_TO_SPACE_ND_H_
#define __HOLIDAY_BATCH_TO_SPACE_ND_H_

#include "HolidayBaseLayer.h"

#include "orz/sync/shotgun.h"
#include "orz/tools/ctxmgr_lite.h"
#include "orz/tools/box.h"

template <class	T>
class HolidayBatchToSpaceNDCPU : public HolidayBaseLayer<T>
{
public:
    HolidayBatchToSpaceNDCPU();
	~HolidayBatchToSpaceNDCPU();

	int Init(Holiday_LayerParameter& inputparam, HolidayNetResource<T> *pdjNetResource);
	
	int Process(std::vector<HolidayFeatureMap<T>*> input_data_map, std::vector<HolidayFeatureMap<T>*>& output_data_map);

    void CaculateOutputSize(int input_number, int input_height, int input_width, int input_channels,
                            int &output_number, int &output_height, int &output_width, int &output_channels);

    void CaculateOutputSize(const std::vector<int> &input_shape, std::vector<int> &output_shape) {
        output_shape.resize(4);
        CaculateOutputSize(input_shape[0], input_shape[2], input_shape[3], input_shape[1],
                           output_shape[0], output_shape[2], output_shape[3], output_shape[1]);
    }
public:
    std::vector<int> m_block_shape;
    std::vector<int> m_crops;
};


template <class	T>
HolidayBatchToSpaceNDCPU<T>::HolidayBatchToSpaceNDCPU()
{

}


template <class	T>
HolidayBatchToSpaceNDCPU<T>::~HolidayBatchToSpaceNDCPU()
{

}

template <class	T>
void HolidayBatchToSpaceNDCPU<T>::CaculateOutputSize(int input_number, int input_height, int input_width, int input_channels,
                        int &output_number, int &output_height, int &output_width, int &output_channels)
{
//    output_number = input_number * m_block_shape[0] * m_block_shape[1];
//    output_height = (input_height + m_paddings[0] + m_paddings[1]) / m_block_shape[0];
//    output_width = (input_width + m_paddings[2] + m_paddings[3]) / m_block_shape[1];
//    output_channels = input_channels;
    output_number = input_number / (m_block_shape[0] * m_block_shape[1]);
    output_height = input_height * m_block_shape[0] - m_crops[0] - m_crops[1];
    output_width = input_width * m_block_shape[1] - m_crops[2] - m_crops[3];
    output_channels = input_channels;
}


template <class	T>
int HolidayBatchToSpaceNDCPU<T>::Init(Holiday_LayerParameter& inputparam, HolidayNetResource<T> *pNetResource)
{
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

	// set top size
	this->top_data_size.resize(1);
	this->top_data_size[0].data_dim.resize(4);
    CaculateOutputSize(this->bottom_data_size[0].data_dim, this->top_data_size[0].data_dim);

	return 0;
}


template <class	T>
int HolidayBatchToSpaceNDCPU<T>::Process(std::vector<HolidayFeatureMap<T>*> input_data_map, std::vector<HolidayFeatureMap<T>*>& output_data_map)
{
	// trans param to cpu
	input_data_map[0]->TransFormDataIn();

	// input data pointer input_data_map[0]->m_cpu.dataMemoryPtr()
	// output data pointer output_data_map[0]->m_cpu.dataMemoryPtr()

	// input data shape input_data_map[0]->data_shape

//	int all_size = 1;
//	for (int i = 0; i < 4; i++)
//	{
//		all_size *= input_data_map[0]->data_shape[i];
//	}
//
//	if (this->bottom_index[0] != this->top_index[0])
//	{
//		memcpy(output_data_map[0]->m_cpu.dataMemoryPtr(), input_data_map[0]->m_cpu.dataMemoryPtr(), sizeof(T)*all_size);
//	}
//	T* start_point = output_data_map[0]->m_cpu.dataMemoryPtr();
//	for (int i = 0; i < all_size; i++)
//	{
//		T val = *start_point;
//		T result = std::max(val, T(0.0)) + m_negetive_slope * std::min(val, T(0.0));
//		*start_point = result;
//		start_point++;
//	}


	// set output data type and shape
	output_data_map[0]->dwStorageType = DATA_CPU_WIDTH;

    // set output data shape
    CaculateOutputSize(input_data_map[0]->data_shape, output_data_map[0]->data_shape);

    // write output
    int input_number = input_data_map[0]->data_shape[0];
    int input_channels = input_data_map[0]->data_shape[1];
    int input_height = input_data_map[0]->data_shape[2];
    int input_width = input_data_map[0]->data_shape[3];

    int output_number = output_data_map[0]->data_shape[0];
    int output_channels = output_data_map[0]->data_shape[1];
    int output_height = output_data_map[0]->data_shape[2];
    int output_width = output_data_map[0]->data_shape[3];

    int input_number_step = input_channels * input_height * input_width;
    int input_channels_step = input_height * input_width;
    int input_height_step = input_width;
    int input_width_step = 1;

    int output_size = output_number * output_channels * output_height * output_width;
    int output_number_step = output_channels * output_height * output_width;
    int output_channels_step = output_height * output_width;
    int output_height_step = output_width;
    int output_width_step = 1;

    const T *input_data = input_data_map[0]->m_cpu.dataMemoryPtr();
    T *output_data = output_data_map[0]->m_cpu.dataMemoryPtr();
//
//    std::memset(output_data, 0, output_size * sizeof(T));
//
    const std::vector<int> &B = m_block_shape;
	const std::vector<int> &C = m_crops;

	auto gun = orz::ctx::lite::ptr<orz::Shotgun>();

	if (gun == nullptr || gun->size() <= 1)
	{
		for (int n = 0; n < output_number; ++n) {
			for (int c = 0; c < output_channels; ++c) {
				for (int h = 0; h < output_height; ++h) {
					for (int w = 0; w < output_width; ++w) {
						int in = ((h + C[0]) % B[0] * B[1] + (w + C[2]) % B[1]) * output_number + n;
						int ic = c;
						int ih = (h + C[0]) / B[0];
						int iw = (w + C[2]) / B[1];

						int at_input_i = in * input_number_step
							+ ic * input_channels_step
							+ ih * input_height_step
							+ iw;

						int at_output_i = n * output_number_step
							+ c * output_channels_step
							+ h * output_height_step
							+ w;

						output_data[at_output_i] = input_data[at_input_i];
					}
				}
			}
		}
	}
	else
	{
		for (int n = 0; n < output_number; ++n) {
			auto bins = orz::split_bins(0, output_channels, int(gun->size()));
			for (auto &bin : bins) {
				gun->fire([&, n, bin](int){
					for (int c = bin.first; c < bin.second; ++c) {
						for (int h = 0; h < output_height; ++h) {
							for (int w = 0; w < output_width; ++w) {
								int in = ((h + C[0]) % B[0] * B[1] + (w + C[2]) % B[1]) * output_number + n;
								int ic = c;
								int ih = (h + C[0]) / B[0];
								int iw = (w + C[2]) / B[1];

								int at_input_i = in * input_number_step
									+ ic * input_channels_step
									+ ih * input_height_step
									+ iw;

								int at_output_i = n * output_number_step
									+ c * output_channels_step
									+ h * output_height_step
									+ w;

								output_data[at_output_i] = input_data[at_input_i];
							}
						}
					}
				});
			}
		}
		gun->join();
	}

	return 0;
}

#endif