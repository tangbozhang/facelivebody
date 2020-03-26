#ifndef __HOLIDAY_REAL_MUL_CPU_H__
#define __HOLIDAY_REAL_MUL_CPU_H__
#include "HolidayBaseLayer.h"



template <class	T>
class HolidayRealMulCPU :public HolidayBaseLayer<T>
{
public:
    HolidayRealMulCPU();
    ~HolidayRealMulCPU();


	int Init(Holiday_LayerParameter& inputparam, HolidayNetResource<T> *pNetResource);
	int Process(std::vector<HolidayFeatureMap<T>*> input_data_map, std::vector<HolidayFeatureMap<T>*>& output_data_map);
public:

	std::vector<int> m_y_shape;
	std::shared_ptr<T> m_y_data;

};


template <class	T>
HolidayRealMulCPU<T>::HolidayRealMulCPU()
{

}

template <class	T>
HolidayRealMulCPU<T>::~HolidayRealMulCPU()
{

}

template <class	T>
int HolidayRealMulCPU<T>::Init(Holiday_LayerParameter& inputparam, HolidayNetResource<T> *pNetResource)
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
    m_y_data.reset(new T[length_y], std::default_delete<T[]>());
    for (int i = 0; i < length_y; i++)
	{
        auto tmp_y_value = blob_y.data(i);
        if (tmp_y_value < FLT_EPSILON && -tmp_y_value < FLT_EPSILON) tmp_y_value = 0;
        m_y_data.get()[i] = tmp_y_value;
	}

	int index = inputparam.bottom_index(0);
	this->bottom_data_size.resize(1);
	this->bottom_data_size[0] = pNetResource->feature_vector_size[index];
	//this->bottom_data_size = inputparam.bottom_data_size;

	this->top_data_size.resize(1);
	this->top_data_size[0] = this->bottom_data_size[0];

	return 0;
}

template <class	T>
int HolidayRealMulCPU<T>::Process(std::vector<HolidayFeatureMap<T>*> input_data_map, std::vector<HolidayFeatureMap<T>*>& output_data_map)
{
	input_data_map[0]->TransFormDataIn();
	if (this->bottom_index[0] != this->top_index[0])
	{
		output_data_map[0]->data_shape = input_data_map[0]->data_shape;
		memcpy(output_data_map[0]->m_cpu.dataMemoryPtr(), input_data_map[0]->m_cpu.dataMemoryPtr(), sizeof(T)*output_data_map[0]->count());
	}
    
    int number = input_data_map[0]->data_shape[0];
    int channels = input_data_map[0]->data_shape[1];
    int height = input_data_map[0]->data_shape[2];
    int width = input_data_map[0]->data_shape[3];
	auto data = output_data_map[0]->m_cpu.dataMemoryPtr();

    int count_y = m_y_shape[0] * m_y_shape[1] * m_y_shape[2] * m_y_shape[3];
    if (count_y == 1)
    {
		T y = *m_y_data.get();
		if (y != T(1))
		{
			T *pstart = output_data_map[0]->m_cpu.dataMemoryPtr();
	        int count = number * channels * height * width;
	        each_do<T>(pstart, count, [&](T &val)
	        {
				val *= y;
			});
		}
    }
	else
	{
		auto gun = orz::ctx::lite::ptr<orz::Shotgun>();
		if (gun == nullptr || gun->size() <= 1)
		{
			T *pstart = output_data_map[0]->m_cpu.dataMemoryPtr();
			for (int n = 0; n < number; n++)
			{
				for (int c = 0; c < channels; c++)
				{
					for (int h = 0; h < height; h++)
					{
						for (int w = 0; w < width; w++)
						{
							auto yn = n % m_y_shape[0];
							auto yc = c % m_y_shape[1];
							auto yh = h % m_y_shape[2];
							auto yw = w % m_y_shape[3];

							*pstart *= m_y_data.get()[((yn * m_y_shape[1] + yc) * m_y_shape[2] + yh) * m_y_shape[3] + yw];

							++pstart;
						}
					}
				}
			}
		}
		else
		{
			auto col_size = width * height;
			auto batch_size = channels * col_size;
			for (int n = 0; n < number; n++)
			{
				auto local_pstart = data + n * batch_size;
				auto bins = orz::split_bins(0, channels, int(gun->size()));
				for (auto &bin : bins)
				{
					gun->fire([&, local_pstart, bin](int) {
						auto pstart = local_pstart + bin.first * col_size;
						for (int c = bin.first; c < bin.second; ++c)
						{
							for (int h = 0; h < height; h++)
							{
								for (int w = 0; w < width; w++)
								{
									auto yn = n % m_y_shape[0];
									auto yc = c % m_y_shape[1];
									auto yh = h % m_y_shape[2];
									auto yw = w % m_y_shape[3];

									*pstart *= m_y_data.get()[((yn * m_y_shape[1] + yc) * m_y_shape[2] + yh) * m_y_shape[3] + yw];

									++pstart;
								}
							}
						}
					});
				}
			}
			gun->join();
		}
	}

	output_data_map[0]->dwStorageType = DATA_CPU_WIDTH;
	output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];

	output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];
	output_data_map[0]->data_shape[1] = input_data_map[0]->data_shape[1];
	output_data_map[0]->data_shape[2] = input_data_map[0]->data_shape[2];
	output_data_map[0]->data_shape[3] = input_data_map[0]->data_shape[3];

	return 0;
}


#endif //!__HOLIDAY_REAL_MUL_CPU_H__
