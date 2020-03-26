#ifndef _HOLIDAY_INNERPRODUCT_CPU_H__
#define _HOLIDAY_INNERPRODUCT_CPU_H__

#include "HolidayBaseLayer.h"
#include "HolidayNetResource.h"
#include "HolidayCommonfuction.h"
#include <cfloat>

#undef USING_SIMOD

template <class	T>
class HolidayInnerProductCPU :public HolidayBaseLayer<T>
{
public:
	HolidayInnerProductCPU();
	~HolidayInnerProductCPU();

	int Init(Holiday_LayerParameter& inputparam, HolidayNetResource<T> *pdjNetResource);
	
	int Process(std::vector<HolidayFeatureMap<T>*> input_data_map, std::vector<HolidayFeatureMap<T>*>& output_data_map);
public:
	std::vector<T> m_bias_value;
	
	HolidayBlobCpu<T>* m_p_inner_blob;
	T* m_innerproduct_width;
	int K_;
	int M_;
	int N_;

	bool transpose_;
private:
private:
	HolidayNetResource<T> * m_p_holiday_net_resource;

};



template <class	T>
HolidayInnerProductCPU<T>::~HolidayInnerProductCPU()
{
	//if (m_innerproduct_width)
	//{
	//	delete[] m_innerproduct_width;
	//}
}

template <class	T>
HolidayInnerProductCPU<T>::HolidayInnerProductCPU()
{
	//m_innerproduct_width = nullptr;
}

template <class	T>
int HolidayInnerProductCPU<T>::Init(Holiday_LayerParameter& inputparam, HolidayNetResource<T> *pNetResource)
{
	this->m_layer_index = inputparam.layer_index();
	m_p_holiday_net_resource = pNetResource;
	int bottom_length = inputparam.bottom_index().size();
	this->bottom_data_size.resize(bottom_length);
	for (size_t i = 0; i < bottom_length; i++)
	{
		int index = inputparam.bottom_index(i);
		this->bottom_data_size[i] = pNetResource->feature_vector_size[index];
	}

	m_bias_value.clear();
	
	const Holiday_BlobProto& bias = inputparam.inner_product_param().bias_param();
	for (int i = 0; i < bias.data_size();i++)
	{
		auto temp_biasvalue = bias.data(i);
		if (temp_biasvalue < FLT_EPSILON && -temp_biasvalue < FLT_EPSILON) temp_biasvalue = 0;
		m_bias_value.push_back(temp_biasvalue);
	}
	//m_bias_value.push_back();
	const Holiday_BlobProto& inner_param = inputparam.inner_product_param().inner_param();
	int index_tmp = 0;
	int all_inner_counts = 1;
	std::vector<int> tmp_shape;
	tmp_shape.resize(inner_param.shape().dim().size());
	for (int i = 0; i <inner_param.shape().dim().size(); i++)
	{
		all_inner_counts *= inner_param.shape().dim(i);
		tmp_shape[i] = inner_param.shape().dim(i);
	}
	N_ = inner_param.shape().dim(0);
	K_ = inner_param.shape().dim(1);

	int index_key = this->m_layer_index;
	if (pNetResource->m_shared_param->param_map.find(index_key) != pNetResource->m_shared_param->param_map.end())
	{

	}
	else
	{
		HolidayBlobCpu<T> tmp_kernel_blob;

		pNetResource->m_shared_param->param_map.insert(std::pair<int, HolidayBlobCpu<T>>(index_key, tmp_kernel_blob));
		pNetResource->m_shared_param->param_map[index_key].Reshape(tmp_shape);

		T* temp_shared_kernel_value = pNetResource->m_shared_param->param_map[index_key].dataMemoryPtr();
		
#ifdef USING_SIMOD 

		for (size_t i = 0; i < N_; i++)
		{
			for (size_t j = 0; j < K_; j++)
			{
				float tmp_float_value = inner_param.data(N_*j + i);
				if (tmp_float_value < FLT_EPSILON && -tmp_float_value < FLT_EPSILON) tmp_float_value = 0;
				temp_shared_kernel_value[K_*i + j] = tmp_float_value;
			}
		}

#else
		for (int i = 0; i < pNetResource->m_shared_param->param_map[index_key].count(); i++)
		{
			float tmp_float_value = inner_param.data(i);
			if (tmp_float_value < FLT_EPSILON && -tmp_float_value < FLT_EPSILON) tmp_float_value = 0;
			*temp_shared_kernel_value = tmp_float_value;
			temp_shared_kernel_value++;
		}
#endif
	}

	m_p_inner_blob = &(pNetResource->m_shared_param->param_map[index_key]);

	//int length2 = inner_param.data_size();
	//m_innerproduct_width = new T[all_inner_counts];
	

	

//#ifdef USING_SIMOD 
//
//	for (size_t i = 0; i < N_; i++)
//	{
//		for (size_t j = 0; j < K_; j++)
//		{
//			m_innerproduct_width[K_*i + j] = inner_param.data(N_*j + i);
//		}
//	}
//	
//#else
//	for (size_t i = 0; i < all_inner_counts; i++)
//	{
//		m_innerproduct_width[index_tmp] = inner_param.data(i);
//		index_tmp++;
//	}
//#endif
	transpose_ = inputparam.inner_product_param().transpose();

	this->top_data_size.resize(1);
	this->top_data_size[0].data_dim.resize(4);
	this->top_data_size[0].data_dim[0] = pNetResource->max_batch_size;
	this->top_data_size[0].data_dim[2] = 1;
	this->top_data_size[0].data_dim[3] = 1;
	this->top_data_size[0].data_dim[1] = inner_param.shape().dim(0);

	return 0;
}

template <class	T>
int HolidayInnerProductCPU<T>::Process(std::vector<HolidayFeatureMap<T>*> input_data_map, std::vector<HolidayFeatureMap<T>*>& output_data_map)
{
	input_data_map[0]->TransFormDataIn();

	output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];
	output_data_map[0]->data_shape[1] = this->top_data_size[0].data_dim[1];
	output_data_map[0]->data_shape[2] = this->top_data_size[0].data_dim[2];
	output_data_map[0]->data_shape[3] = this->top_data_size[0].data_dim[3];

	//T*	pstart = input_data_map[0]->m_data1.dataMemoryPtr();
	//const T* weight = m_innerproduct_width;
	const T* weight = m_p_inner_blob->dataMemoryPtr();
	const T* bottom_data = input_data_map[0]->m_cpu.dataMemoryPtr();
	T* top_data = output_data_map[0]->m_cpu.dataMemoryPtr();
	//int single_size_input = input_data_map[0]->m_data1.n_single_image_size;
	//int single_size_output = input_data_map[0]->m_data1.n_single_image_size;

	M_ = input_data_map[0]->data_shape[0];

	bool fuse_bias = !m_bias_value.empty();
	if (fuse_bias)
	{
		SetBiasBlob(output_data_map[0]->m_cpu, output_data_map[0]->data_shape, m_bias_value);
	}
	
#ifdef USING_SIMOD	
	caffe_cpu_gemm_simod(false, true, M_,N_ , K_,
		bottom_data, weight, top_data);
#else
	caffe_cpu_gemm<T>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
		M_, N_, K_, (T)1.,
		bottom_data, weight, fuse_bias ? (T)1. : (T)0., top_data);


#endif
	// if (!m_bias_value.empty())
	// {
	// 	AddBiasBlob(output_data_map[0]->m_cpu, output_data_map[0]->data_shape, m_bias_value);
	// }

	output_data_map[0]->dwStorageType = DATA_CPU_WIDTH;

	output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];
	output_data_map[0]->data_shape[1] = this->top_data_size[0].data_dim[1];
	output_data_map[0]->data_shape[2] = this->top_data_size[0].data_dim[2];
	output_data_map[0]->data_shape[3] = this->top_data_size[0].data_dim[3];

	return 0;
}


#endif //!_INNERPRODUCT_H__
