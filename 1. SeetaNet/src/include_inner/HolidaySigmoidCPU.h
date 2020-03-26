#ifndef __HOLIDAY_SIGMOID_CPU_H__
#define __HOLIDAY_SIGMOID_CPU_H__
#include "HolidayBaseLayer.h"


template <class	T>
class HolidaySigmoidCPU : public HolidayBaseLayer<T>
{
public:
	HolidaySigmoidCPU();
	~HolidaySigmoidCPU();

	int Init(Holiday_LayerParameter& inputparam, HolidayNetResource<T> *pNetResource);
	int Process(std::vector<HolidayFeatureMap<T>*> input_data_map, std::vector<HolidayFeatureMap<T>*>& output_data_map);

};

template <class	T>
HolidaySigmoidCPU<T>::HolidaySigmoidCPU()
{

}


template <class	T>
HolidaySigmoidCPU<T>::~HolidaySigmoidCPU()
{

}


template <class	T>
int HolidaySigmoidCPU<T>::Init(Holiday_LayerParameter& inputparam, HolidayNetResource<T> *pNetResource)
{
	/*input_rows = inputparam.param_vector[0];
	input_cols = inputparam.param_vector[1];
	input_slices = inputparam.param_vector[2];*/
	int index = inputparam.bottom_index(0);
	this->bottom_data_size.resize(1);
	this->bottom_data_size[0] = pNetResource->feature_vector_size[index];
	//this->bottom_data_size = inputparam.bottom_data_size;

	this->top_data_size.resize(1);
	this->top_data_size[0] = this->bottom_data_size[0];

	return 0;
}

template <typename Dtype>
inline Dtype sigmoid_funtion(Dtype x) {
	return 1. / (1. + exp(-x));
}

template<typename T>
static void sigmoid_each(T *arr, size_t size) {
	auto gun = orz::ctx::lite::ptr<orz::Shotgun>();
	if (gun == nullptr || gun->size() <= 1)
	{
		for (size_t i = 0; i < size; ++i)
		{
			*arr = sigmoid_funtion<T>(*arr);
			++arr;
		}
	}
	else
	{
		auto bins = orz::lsplit_bins(0, size, gun->size());
		for (auto &bin : bins) {
			gun->fire([&, bin](int){
				auto local_arr = arr + bin.first;
				for (size_t i = bin.first; i < bin.second; ++i)
				{
					*local_arr = sigmoid_funtion<T>(*local_arr);
					++local_arr;
				}
			});
		}
		gun->join();
	}
}



template <class	T>
int HolidaySigmoidCPU<T>::Process(std::vector<HolidayFeatureMap<T>*> input_data_map, std::vector<HolidayFeatureMap<T>*>& output_data_map)
{
	input_data_map[0]->TransFormDataIn();
	if (this->bottom_index[0] != this->top_index[0])
	{
		output_data_map[0]->dwStorageType = DATA_CPU_WIDTH;
		output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];

		output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];
		output_data_map[0]->data_shape[1] = input_data_map[0]->data_shape[1];
		output_data_map[0]->data_shape[2] = input_data_map[0]->data_shape[2];
		output_data_map[0]->data_shape[3] = input_data_map[0]->data_shape[3];

		memcpy(output_data_map[0]->m_cpu.dataMemoryPtr(), input_data_map[0]->m_cpu.dataMemoryPtr(), sizeof(T)*output_data_map[0]->count());
	}

	sigmoid_each<T>(output_data_map[0]->m_cpu.dataMemoryPtr(), output_data_map[0]->count());

	return 0;
}

#endif //!__SIGMOID_H__