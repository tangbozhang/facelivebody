#pragma once
#include "HolidayBaseLayer.h"
#include "HolidayBlobCpu.h"
#include "HolidayCommonfuction.h"

template<typename T>
class HolidayLocalResponseNormalizeCpu :
	public HolidayBaseLayer<T>
{
public:
	HolidayLocalResponseNormalizeCpu();
	~HolidayLocalResponseNormalizeCpu();

	int Init(Holiday_LayerParameter & inputparam, HolidayNetResource<T> * p_holiday_net_resource);
	int Process(std::vector<HolidayFeatureMap<T>*> input_data_map, std::vector<HolidayFeatureMap<T>*>&output_data_map);
private:

	int CrossChannelProcess(std::vector<HolidayFeatureMap<T>*> input_data_map, std::vector<HolidayFeatureMap<T>*> output_data_map);

	T alpha_;
	T beta_;
	int64_t localsize_;
	int64_t k_;
	int32_t normalize_type;

	T zero_minus_beta;


	T cross_channel_multi_scale;
	int cross_channel_start_pos;
	int cross_channel_copy_channel;
	HolidayBlobCpu<T> cross_channel_squart_cube;
	HolidayBlobCpu<T> cross_sum_cube;
	HolidayBlobCpu<T> cross_scale_cube;
	HolidayBlobCpu<T> cross_output_cube;

	int channels_;
	int height_;
	int width_;
};

template <class	T>
int HolidayLocalResponseNormalizeCpu<T>::CrossChannelProcess(std::vector<HolidayFeatureMap<T>*> input_data_map, std::vector<HolidayFeatureMap<T>*> output_data_map)
{
	input_data_map[0]->TransFormDataIn();
	T* padded_square_data = cross_channel_squart_cube.dataMemoryPtr();
	T* scale_data = cross_scale_cube.dataMemoryPtr();
	T* top_data = output_data_map[0]->m_cpu.dataMemoryPtr();
	const T* bottom_data = input_data_map[0]->m_cpu.dataMemoryPtr();


	for (int n = 0; n < input_data_map[0]->data_shape[0]; ++n) {
		for (int i = 0; i < cross_scale_cube.count(); ++i) {
			cross_scale_cube.dataMemoryPtr()[i] = k_;
		}
		caffe_set(cross_channel_squart_cube.count(), T(0), padded_square_data);
		// compute the padded square
		caffe_sqr(channels_ * height_ * width_,
			bottom_data + input_data_map[0]->m_cpu.offset(n),
			padded_square_data + cross_channel_squart_cube.offset(0, cross_channel_start_pos));
		// Create the first channel scale
		for (int c = 0; c < localsize_; ++c) {
			caffe_axpy<T>(height_ * width_, cross_channel_multi_scale,
				padded_square_data + cross_channel_squart_cube.offset(0, c),
				scale_data + cross_scale_cube.offset(0, 0));
		}
		for (int c = 1; c < channels_; ++c) {
			// copy previous scale
			caffe_copy<T>(height_ * width_,
				scale_data + cross_scale_cube.offset(0, c - 1),
				scale_data + cross_scale_cube.offset(0, c));
			// add head
			caffe_axpy<T>(height_ * width_, cross_channel_multi_scale,
				padded_square_data + cross_channel_squart_cube.offset(0, c + localsize_ - 1),
				scale_data + cross_scale_cube.offset(0, c));
			// subtract tail
			caffe_axpy<T>(height_ * width_, -cross_channel_multi_scale,
				padded_square_data + cross_channel_squart_cube.offset(0, c - 1),
				scale_data + cross_scale_cube.offset(0, c));
		}
		caffe_powx<T>(cross_scale_cube.count(), scale_data, -beta_, top_data);
		caffe_mul<T>(cross_scale_cube.count(), top_data, bottom_data + input_data_map[0]->m_cpu.offset(n), top_data);
		top_data += input_data_map[0]->m_cpu.offset(1);
	}


	return 0;
}

template <class	T>
HolidayLocalResponseNormalizeCpu<T>::HolidayLocalResponseNormalizeCpu()
{
	cross_channel_start_pos = 0;
	cross_channel_copy_channel = 0;
}


template <class	T>
HolidayLocalResponseNormalizeCpu<T>::~HolidayLocalResponseNormalizeCpu()
{

}


template<typename T>
int HolidayLocalResponseNormalizeCpu<T>::Process(std::vector<HolidayFeatureMap<T>*> input_data_map, std::vector<HolidayFeatureMap<T>*>&output_data_map)
{
	if (1 != (localsize_ % 2))
	{
		return -1;
	}

	//output_data_map[0]->m_data2 = input_data_map[0]->m_data2;
	//output_data_map[0] = &m_data_output;
	switch (normalize_type)
	{
	case 0:
	{
		CrossChannelProcess(input_data_map, output_data_map);
		break;
	}
	case 1:
	{
		break;
	}
	default:
	{

	}
	};
	output_data_map[0]->dwStorageType = DATA_CPU_WIDTH;
	output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];

	output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];
	output_data_map[0]->data_shape[1] = this->top_data_size[0].data_dim[1];
	output_data_map[0]->data_shape[2] = this->top_data_size[0].data_dim[2];
	output_data_map[0]->data_shape[3] = this->top_data_size[0].data_dim[3];
	return 0;
}

template<typename T>
int HolidayLocalResponseNormalizeCpu<T>::Init(Holiday_LayerParameter & inputparam, HolidayNetResource<T> * p_holiday_net_resource)
{
	int bottom_index = inputparam.bottom_index(0);
	HolidayDataSize bottom_size = p_holiday_net_resource->feature_vector_size[bottom_index];
	this->bottom_data_size.resize(1);
	this->bottom_data_size[0]= bottom_size;
	localsize_ = inputparam.lrn_param().local_size();
	alpha_ = inputparam.lrn_param().alpha();
	beta_ = inputparam.lrn_param().beta();
	k_ = inputparam.lrn_param().k();
	normalize_type = inputparam.lrn_param().norm_region();

	std::vector<int> scale_shape_vector = this->bottom_data_size[0].data_dim;
	scale_shape_vector[0] = 1;
	cross_scale_cube.Reshape(scale_shape_vector);

	std::vector<int> cross_channel_squart_cube_shape = scale_shape_vector;
	cross_channel_squart_cube_shape[1] += localsize_ - 1;

	cross_channel_squart_cube.Reshape(cross_channel_squart_cube_shape);

	cross_channel_start_pos = (localsize_ - 1) / 2;
	cross_channel_copy_channel = this->bottom_data_size[0].data_dim[1];

	this->top_data_size.resize(1);
	this->top_data_size[0] = this->bottom_data_size[0];

	cross_channel_multi_scale = alpha_ / localsize_;

	zero_minus_beta = 0 - beta_;



	channels_ = this->bottom_data_size[0].data_dim[1];
	height_ = this->bottom_data_size[0].data_dim[2];
	width_ = this->bottom_data_size[0].data_dim[3];

	return 0;
}

