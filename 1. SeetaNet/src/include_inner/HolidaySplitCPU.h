#ifndef __HOLIDAY_SPLIT_CPU_H__
#define __HOLIDAY_SPLIT_CPU_H__
#include "HolidayBaseLayer.h"

template <class	T>
class HolidaySplitCPU :public HolidayBaseLayer<T>
{
public:
	HolidaySplitCPU();
	~HolidaySplitCPU();

	 int Init(Holiday_LayerParameter& inputparam, HolidayNetResource<T> *pNetResource);
	 int Process(std::vector<HolidayFeatureMap<T>*> input_data_map, std::vector<HolidayFeatureMap<T>*>& output_data_map);
	
public:

};

template <class	T>
HolidaySplitCPU<T>::HolidaySplitCPU()
{

}


template <class	T>
HolidaySplitCPU<T>::~HolidaySplitCPU()
{

}


template <class	T>
int HolidaySplitCPU<T>::Init(Holiday_LayerParameter& inputparam, HolidayNetResource<T> *pNetResource)
{
	int index = inputparam.bottom_index(0);
	this->bottom_data_size.resize(1);
	this->bottom_data_size[0] = pNetResource->feature_vector_size[index];
	
	//this->F = inputparam.bottom_data_size;
	this->top_data_size.resize(inputparam.top_index().size());
	for (int i = 0; i < inputparam.top_index().size(); i++)
	{
		this->top_data_size[i] = this->bottom_data_size[0];
	}

	return 0;
}

template <class	T>
int HolidaySplitCPU<T>::Process(std::vector<HolidayFeatureMap<T>*> input_data_map, std::vector<HolidayFeatureMap<T>*>& output_data_map)
{
	input_data_map[0]->TransFormDataIn();
	int all_size = 1;
	for (int i = 0; i < 4; i++)
	{
		all_size *= input_data_map[0]->data_shape[i];
	}
	for (int i = 0; i < this->top_index.size(); i++)
	{
		memcpy(output_data_map[i]->m_cpu.dataMemoryPtr(), input_data_map[0]->m_cpu.dataMemoryPtr(), sizeof(T)*all_size);
		output_data_map[i]->data_shape[0] = input_data_map[0]->data_shape[0];
		output_data_map[i]->dwStorageType = DATA_CPU_WIDTH;

		output_data_map[i]->data_shape[0] = input_data_map[0]->data_shape[0];
		output_data_map[i]->data_shape[1] = input_data_map[0]->data_shape[1];
		output_data_map[i]->data_shape[2] = input_data_map[0]->data_shape[2];
		output_data_map[i]->data_shape[3] = input_data_map[0]->data_shape[3];
	}


	return 0;
}

#endif //!__SPLIT_H__
