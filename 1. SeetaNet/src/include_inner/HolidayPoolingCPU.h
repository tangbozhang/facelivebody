#ifndef _HOLIDAY_POOLING_H__
#define _HOLIDAY_POOLING_H__

#include "HolidayBaseLayer.h"

#include "orz/sync/shotgun.h"
#include "orz/tools/ctxmgr_lite.h"
#include "orz/tools/box.h"

template <typename T>
class HolidayPoolingCpu :public HolidayBaseLayer<T>
{
public:
	HolidayPoolingCpu();
	~HolidayPoolingCpu();
	int Init(Holiday_LayerParameter& inputparam, HolidayNetResource<T> *pdjNetResource);
	int Process(std::vector<HolidayFeatureMap<T>*> input_data_map, std::vector<HolidayFeatureMap<T>*>& output_data_map);

	int MaxPooling(int number, HolidayBlobCpu<T>& inputdata, HolidayBlobCpu<T>& outputdata, size_t kernel_h, size_t kernel_w,
		size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w, std::vector<int> &shape_vector_in, std::vector<int>& shape_vector_out);

	int AveragePooling(int number, HolidayBlobCpu<T>& inputdata, HolidayBlobCpu<T>& outputdata, size_t kernel_h, size_t kernel_w,
		size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w, std::vector<int> &shape_vector_in, std::vector<int>& shape_vector_out);

private:
	void CaculatePoolSize(int input_height, int input_width, int &output_height, int &output_width);


	int m_kernel_h;
	int m_kernel_w;
	int m_stride_h;
	int m_stride_w;
	int m_pad_h;
	int m_pad_w;
	int m_dilation_h;
	int m_dilation_w;

	int m_pool_type;

	bool m_valid;

	int m_pooled_height_;
	int	m_pooled_width_;

	std::string m_tf_padding;
	int m_tf_fake_padding_h = 0;
	int m_tf_fake_padding_w = 0;
};

size_t offset(std::vector<int> shape_, const int n, const int c = 0, const int h = 0,
	const int w = 0)
{
	return ((n * shape_[1] + c) * shape_[2] + h) * shape_[3] + w;
}

template <typename T>
void HolidayPoolingCpu<T>::CaculatePoolSize(int input_height, int input_width, int &output_height, int &output_width)
{

	if (m_tf_padding == "VALID")
	{
		output_height = ceil((input_height + 2 * m_pad_h - m_kernel_h + 1) / (float)m_stride_h);
		output_width = ceil((input_width + 2 * m_pad_w - m_kernel_w + 1) / (float)m_stride_w);
	}
	else if (m_tf_padding == "SAME")
	{
		output_height = ceil((input_height + 2 * m_pad_h) / (float)m_stride_h);
		output_width = ceil((input_width + 2 * m_pad_w) / (float)m_stride_w);

		// no feak padding when pooling
		m_tf_fake_padding_h = 0;
		m_tf_fake_padding_w = 0;
	}
	else if (m_valid)
	{
		output_height = floor((input_height + 2 * m_pad_h - m_kernel_h) / (float)m_stride_h + 1);
		output_width = floor((input_width + 2 * m_pad_w - m_kernel_w) / (float)m_stride_w + 1);
	}
	else
	{
		output_height = ceil((input_height + 2 * m_pad_h - m_kernel_h) / (float)m_stride_h + 1);
		output_width = ceil((input_width + 2 * m_pad_w - m_kernel_w) / (float)m_stride_w + 1);
	}
}


template <typename T>
int HolidayPoolingCpu<T>::Init(Holiday_LayerParameter& inputparam, HolidayNetResource<T> *pNetResource)
{
	//m_pad_h = inputparam.param_vector[6];
	//m_pad_w = inputparam.param_vector[7];
	m_dilation_h = 1;
	m_dilation_w = 1;

	m_pool_type = inputparam.pooling_param().pool();

	m_kernel_h = inputparam.pooling_param().kernel_height();
	m_kernel_w = inputparam.pooling_param().kernel_width();
	m_stride_h = inputparam.pooling_param().stride_height();
	m_stride_w = inputparam.pooling_param().stride_width();
	m_pad_h = inputparam.pooling_param().pad_height();
	m_pad_w = inputparam.pooling_param().pad_width();

	m_valid = false;
	if (inputparam.pooling_param().has_valid())
	{
		m_valid = inputparam.pooling_param().valid();
	}

	if (inputparam.pooling_param().has_tf_padding())
	{
		m_tf_padding = inputparam.pooling_param().tf_padding();
	}

	int bottom_index = inputparam.bottom_index(0);
	HolidayDataSize bottom_size = pNetResource->feature_vector_size[bottom_index];
	this->bottom_data_size.resize(1);
	this->bottom_data_size[0] = bottom_size;

	if (inputparam.pooling_param().global_pooling())
	{
		m_kernel_h = this->bottom_data_size[0].data_dim[2];
		m_kernel_w = this->bottom_data_size[0].data_dim[3];
		m_pad_h = 0;
		m_pad_w = 0;
	}

	CaculatePoolSize(this->bottom_data_size[0].data_dim[2], this->bottom_data_size[0].data_dim[3], m_pooled_height_, m_pooled_width_);

	this->top_data_size.resize(1);
	this->top_data_size[0].data_dim.resize(4);
	this->top_data_size[0].data_dim[2] = m_pooled_height_;
	this->top_data_size[0].data_dim[3] = m_pooled_width_;
	this->top_data_size[0].data_dim[1] = this->bottom_data_size[0].data_dim[1];
	this->top_data_size[0].data_dim[0] = this->bottom_data_size[0].data_dim[0];

	return 0;
}

template <typename T>
int HolidayPoolingCpu<T>::Process(std::vector<HolidayFeatureMap<T>*> input_data_map, std::vector<HolidayFeatureMap<T>*>& output_data_map)
{
	input_data_map[0]->TransFormDataIn();

	CaculatePoolSize(input_data_map[0]->data_shape[2], input_data_map[0]->data_shape[3], m_pooled_height_, m_pooled_width_);

	std::vector<int> shape_vector_in;
	shape_vector_in.push_back(input_data_map[0]->data_shape[0]);
	shape_vector_in.push_back(input_data_map[0]->data_shape[1]);
	shape_vector_in.push_back(input_data_map[0]->data_shape[2]);
	shape_vector_in.push_back(input_data_map[0]->data_shape[3]);

	std::vector<int> shape_vector_out;
	shape_vector_out.push_back(input_data_map[0]->data_shape[0]);
	shape_vector_out.push_back(input_data_map[0]->data_shape[1]);
	shape_vector_out.push_back(m_pooled_height_);
	shape_vector_out.push_back(m_pooled_width_);

	//output_data_map[0] = &m_data_output;
	if (Holiday_PoolingParameter::MAX == m_pool_type)
	{
		//MaxPooling(input_data_map[0]->data_shape[0], input_data_map[0]->m_cpu, output_data_map[0]->m_cpu,
		//	m_kernel_h, m_kernel_w, m_stride_h, m_stride_w, m_pad_h, m_pad_w);

		MaxPooling(input_data_map[0]->data_shape[0], input_data_map[0]->m_cpu, output_data_map[0]->m_cpu,
			m_kernel_h, m_kernel_w, m_stride_h, m_stride_w, m_pad_h + m_tf_fake_padding_h, m_pad_w + m_tf_fake_padding_w, shape_vector_in, shape_vector_out);

	}
	else if (Holiday_PoolingParameter::AVE == m_pool_type)
	{
		//AveragePooling(input_data_map[0]->data_shape[0], input_data_map[0]->m_cpu, output_data_map[0]->m_cpu,
		//	m_kernel_h, m_kernel_w, m_stride_h, m_stride_w, m_pad_h, m_pad_w);

		AveragePooling(input_data_map[0]->data_shape[0], input_data_map[0]->m_cpu, output_data_map[0]->m_cpu,
			m_kernel_h, m_kernel_w, m_stride_h, m_stride_w, m_pad_h + m_tf_fake_padding_h, m_pad_w + m_tf_fake_padding_w, shape_vector_in, shape_vector_out);

	}
	else{}

	output_data_map[0]->dwStorageType = DATA_CPU_WIDTH;
	output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];


	output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];
	output_data_map[0]->data_shape[1] = shape_vector_out[1];
	output_data_map[0]->data_shape[2] = shape_vector_out[2];
	output_data_map[0]->data_shape[3] = shape_vector_out[3];

	return 0;
}


template <typename T>
HolidayPoolingCpu<T>::HolidayPoolingCpu()
{
};
template <typename T>
HolidayPoolingCpu<T>::~HolidayPoolingCpu()
{
};

template <typename T>
int HolidayPoolingCpu<T>::MaxPooling(int number, HolidayBlobCpu<T>& inputdata, HolidayBlobCpu<T>& outputdata,
	size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w
	, std::vector<int> &shape_vector_in, std::vector<int>& shape_vector_out)
{
	//预先计算特征输出向量的维度
	//W2 = (W1−kernerl_w + 2P) / S + 1
	//input_data_map[0]->TransFormDataIn();
	const T* bottom_data = inputdata.dataMemoryPtr();
	T* top_data = outputdata.dataMemoryPtr();
	int height_ = shape_vector_in[2];
	int width_ = shape_vector_in[3];

	int input_offset = offset(shape_vector_in, 0, 1);
	int output_offset = offset(shape_vector_out, 0, 1);

	auto gun = orz::ctx::lite::ptr<orz::Shotgun>();
	if (gun == nullptr || gun->size() <= 1)
	{
		for (int n = 0; n < number; ++n)
		{
			for (int c = 0; c < inputdata.shape()[1]; ++c)
			{
				for (int ph = 0; ph < m_pooled_height_; ++ph)
				{
					for (int pw = 0; pw < m_pooled_width_; ++pw)
					{
						int hstart = ph * stride_h - pad_h;
						int wstart = pw * stride_w - pad_w;
						int hend_tmp = hstart + kernel_h;
						int wend_tmp = wstart + kernel_w;
						int hend = std::min(hend_tmp, height_);
						int wend = std::min(wend_tmp, width_);
						hstart = std::max(hstart, 0);
						wstart = std::max(wstart, 0);
						const int pool_index = ph * m_pooled_width_ + pw;
						T max_value_ = bottom_data[hstart * width_ + wstart];
						for (int h = hstart; h < hend; ++h)
						{
							for (int w = wstart; w < wend; ++w)
							{
								const int index = h * width_ + w;
								if (bottom_data[index] > max_value_)
								{
									max_value_ = bottom_data[index];
								}
							}
						}
						top_data[pool_index] = max_value_;
					}
				}
				// compute offset
				bottom_data += input_offset;
				top_data += output_offset;
			}
		}
	}
	else
	{
		auto input_batch_offset = inputdata.shape()[1] * input_offset;
		auto output_batch_offset = inputdata.shape()[1] * output_offset;
		for (int n = 0; n < number; ++n)
		{
			auto batch_bottom_data = bottom_data + n * input_batch_offset;
			auto batch_top_data = top_data + n * output_batch_offset;
			auto bins = orz::split_bins(0, inputdata.shape()[1], int(gun->size()));
			for (auto &bin : bins)
			{
				gun->fire([&, batch_bottom_data, batch_top_data, bin](int)
				{
					auto local_bottom_data = batch_bottom_data + bin.first * input_offset;
					auto local_top_data = batch_top_data + bin.first * output_offset;
					for (int c = bin.first; c < bin.second; ++c)
					{
						for (int ph = 0; ph < m_pooled_height_; ++ph)
						{
							for (int pw = 0; pw < m_pooled_width_; ++pw)
							{
								int hstart = ph * stride_h - pad_h;
								int wstart = pw * stride_w - pad_w;
								int hend_tmp = hstart + kernel_h;
								int wend_tmp = wstart + kernel_w;
								int hend = std::min(hend_tmp, height_);
								int wend = std::min(wend_tmp, width_);
								hstart = std::max(hstart, 0);
								wstart = std::max(wstart, 0);
								const int pool_index = ph * m_pooled_width_ + pw;
								T max_value_ = local_bottom_data[hstart * width_ + wstart];
								for (int h = hstart; h < hend; ++h)
								{
									for (int w = wstart; w < wend; ++w)
									{
										const int index = h * width_ + w;
										if (local_bottom_data[index] > max_value_)
										{
											max_value_ = local_bottom_data[index];
										}
									}
								}
								local_top_data[pool_index] = max_value_;
							}
						}
						// compute offset
						local_bottom_data += input_offset;
						local_top_data += output_offset;
					}
				});
			}
		}
		gun->join();
	}

	return 0;
};

template <typename T>
int HolidayPoolingCpu<T>::AveragePooling(int number, HolidayBlobCpu<T>& inputdata, HolidayBlobCpu<T>& outputdata,
	size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w
	, std::vector<int> &shape_vector_in, std::vector<int>& shape_vector_out)
{
	//input_data_map[0]->TransFormDataIn();
	const T* bottom_data = inputdata.dataMemoryPtr();
	T* top_data = outputdata.dataMemoryPtr();

	int height_ = shape_vector_in[2];
	int width_ = shape_vector_in[3];

	int input_offset = offset(shape_vector_in, 0, 1);
	int output_offset = offset(shape_vector_out, 0, 1);

	//int  pooled_width_ = ceil((inputdata.n_cols + 2 * pad_w - kernel_w) / (float)stride_w + 1);
	//int  pooled_height_ = ceil((inputdata.n_rows + 2 * pad_h - kernel_h) / (float)stride_h + 1);

	auto gun = orz::ctx::lite::ptr<orz::Shotgun>();
	if (gun == nullptr || gun->size() <= 1)
	{
		for (int n = 0; n < number; ++n)
		{
			for (int c = 0; c < inputdata.shape()[1]; ++c)
			{
				for (int ph = 0; ph < m_pooled_height_; ++ph)
				{
					for (int pw = 0; pw < m_pooled_width_; ++pw)
					{
						int hstart = ph * stride_h - pad_h;
						int wstart = pw * stride_w - pad_w;
						int hend_tmp = hstart + kernel_h;
						int wend_tmp = wstart + kernel_w;
						int hend = std::min(hend_tmp, height_);
						int wend = std::min(wend_tmp, width_);
						hstart = std::max(hstart, 0);
						wstart = std::max(wstart, 0);
						const int pool_index = ph * m_pooled_width_ + pw;
						int current_count = 0;
						T sum_value = 0.0;
						for (int h = hstart; h < hend; ++h)
						{
							for (int w = wstart; w < wend; ++w)
							{
								const int index = h * width_ + w;
								sum_value += bottom_data[index];
								current_count += 1;
							}
						}
						top_data[pool_index] = sum_value / current_count;
					}
				}
				// compute offset
				bottom_data += input_offset;
				top_data += output_offset;
			}
		}
	}
	else
	{
		auto input_batch_offset = inputdata.shape()[1] * input_offset;
		auto output_batch_offset = inputdata.shape()[1] * output_offset;
		for (int n = 0; n < number; ++n)
		{
			auto batch_bottom_data = bottom_data + n * input_batch_offset;
			auto batch_top_data = top_data + n * output_batch_offset;
			auto bins = orz::split_bins(0, inputdata.shape()[1], int(gun->size()));
			for (auto &bin : bins)
			{
				gun->fire([&, batch_bottom_data, batch_top_data, bin](int)
				{
					auto local_bottom_data = batch_bottom_data + bin.first * input_offset;
					auto local_top_data = batch_top_data + bin.first * output_offset;
					for (int c = bin.first; c < bin.second; ++c)
					{
						for (int ph = 0; ph < m_pooled_height_; ++ph)
						{
							for (int pw = 0; pw < m_pooled_width_; ++pw)
							{
								int hstart = ph * stride_h - pad_h;
								int wstart = pw * stride_w - pad_w;
								int hend_tmp = hstart + kernel_h;
								int wend_tmp = wstart + kernel_w;
								int hend = std::min(hend_tmp, height_);
								int wend = std::min(wend_tmp, width_);
								hstart = std::max(hstart, 0);
								wstart = std::max(wstart, 0);
								const int pool_index = ph * m_pooled_width_ + pw;
								int current_count = 0;
								T sum_value = 0.0;
								for (int h = hstart; h < hend; ++h)
								{
									for (int w = wstart; w < wend; ++w)
									{
										const int index = h * width_ + w;
										sum_value += local_bottom_data[index];
										current_count += 1;
									}
								}
								local_top_data[pool_index] = sum_value / current_count;
							}
						}
						// compute offset
						local_bottom_data += input_offset;
						local_top_data += output_offset;
					}
				});
			}
		}
		gun->join();
	}

	return 0;
};

#endif