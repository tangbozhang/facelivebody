#ifndef _HOLIDAYCONVOLUTIONCPU_H__
#define _HOLIDAYCONVOLUTIONCPU_H__

#include <vector>
#include<iomanip>

#include "HolidayBaseLayer.h"
#include "HolidayCommon.h"
#include "HolidayFeatureMap.h"
#include "HolidayNetResource.h"
#include "HolidayCommonfuction.h"
#include "im2col.hpp"
#include "simod_gemm.h"
#include <fstream>
#include <cfloat>

template<class T>
class HolidayConvolutionCPU : public HolidayBaseLayer<T>
{
public:
    HolidayConvolutionCPU();
    HolidayConvolutionCPU(const HolidayConvolutionCPU& m);
    ~HolidayConvolutionCPU();
    int Init(Holiday_LayerParameter & inputparam, HolidayNetResource<T> * p_holiday_net_resource);
    int Process(std::vector<HolidayFeatureMap<T>*> input_data_map, std::vector<HolidayFeatureMap<T>*>&output_data_map);


    int m_stride_h;
    int m_stride_w;
    int m_pad_h;
    int m_pad_w;
    int m_dilation_h;
    int m_dilation_w;
    int m_kenerl_channels;
    int m_kernel_h;
    int m_kernel_w;
    int m_group_;
    int m_kenerl_number;
    int kernel_dims_;
    std::vector<T> m_bias_value;
   
    std::vector<int> col_buffer_shape_;
    //HolidayBlobCpu<T> m_kernel_blob;
    HolidayBlobCpu<T>* m_p_kernel_blob;
    int weight_offset_;
    int conv_out_spatial_dim_;
    int col_offset_;
	int output_offset_;

	std::string m_tf_padding;
	int m_tf_fake_padding_h = 0;
	int m_tf_fake_padding_w = 0;
	int m_tf_conv_shift_h = 0;
    int m_tf_conv_shift_w = 0;
private:
    HolidayNetResource<T> * m_p_holiday_net_resource;

private:
    inline void conv_im2col_cpu(const T* data, T* col_buff)
    {
        shift_im2col_cpu(data, this->bottom_data_size[0].data_dim[1],
            this->bottom_data_size[0].data_dim[2], this->bottom_data_size[0].data_dim[3],
            m_kernel_h, m_kernel_w,
			m_pad_h + m_tf_fake_padding_h, m_pad_w + m_tf_fake_padding_w,
            m_tf_conv_shift_h, m_tf_conv_shift_w,
            m_stride_h, m_stride_w,
            m_dilation_h, m_dilation_w, col_buff);

    }
    int Caculate(const int height, const int width,
        const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int dilation_h, const int dilation_w, int& output_h, int& output_w);

    inline void conv_im2col_cpu(HolidayFeatureMap<T>*input_map, T*data, T* col_buff)
    {
        shift_im2col_cpu(data, input_map->data_shape[1],
            input_map->data_shape[2], input_map->data_shape[3],
            m_kernel_h, m_kernel_w,
            m_pad_h + m_tf_fake_padding_h, m_pad_w + m_tf_fake_padding_w,
            m_tf_conv_shift_h, m_tf_conv_shift_w,
            m_stride_h, m_stride_w,
            m_dilation_h, m_dilation_w, col_buff);

    }

    

};

template<class T>
int HolidayConvolutionCPU<T>::Init(Holiday_LayerParameter & inputparam, HolidayNetResource<T> * p_holiday_net_resource)
{
    this->m_layer_index = inputparam.layer_index();
    m_p_holiday_net_resource = p_holiday_net_resource;
    int bottom_index = inputparam.bottom_index(0);
    HolidayDataSize bottom_size = p_holiday_net_resource->feature_vector_size[bottom_index];
    this->bottom_data_size.resize(1);
    this->bottom_data_size[0] = bottom_size;

    std::vector<int> shape;
    const ::Holiday_BlobShape& tmp_shape = inputparam.convolution_param().kernel_param().shape();

    for (int i = 0; i < tmp_shape.dim_size(); i++)
    {
        shape.push_back(tmp_shape.dim(i));
    }
    
    //m_kernel_blob.Reshape(shape);
    int index_key = this->m_layer_index;
    if (p_holiday_net_resource->m_shared_param->param_map.find(index_key) != p_holiday_net_resource->m_shared_param->param_map.end())
    {
        
    }
    else
    {
        HolidayBlobCpu<T> tmp_kernel_blob;
        
        p_holiday_net_resource->m_shared_param->param_map.insert(std::pair<int, HolidayBlobCpu<T>> ( index_key, tmp_kernel_blob));
        p_holiday_net_resource->m_shared_param->param_map[index_key].Reshape(shape);
    
        T* temp_shared_kernel_value = p_holiday_net_resource->m_shared_param->param_map[index_key].dataMemoryPtr();
        for (int i = 0; i < p_holiday_net_resource->m_shared_param->param_map[index_key].count(); i++)
        {
            float tmp_float_value = inputparam.convolution_param().kernel_param().data(i);
			if (tmp_float_value < FLT_EPSILON && -tmp_float_value < FLT_EPSILON) tmp_float_value = 0;
            *temp_shared_kernel_value = tmp_float_value;
            temp_shared_kernel_value++;
        }
    }
    m_p_kernel_blob = &(p_holiday_net_resource->m_shared_param->param_map[index_key]);

    //T* temp_kernel_value = m_kernel_blob.dataMemoryPtr();
    //for (int i = 0; i < m_kernel_blob.count_; i++)
    //{
    //    float tmp_float_value = inputparam.convolution_param().kernel_param().data(i);
    //    *temp_kernel_value = tmp_float_value;
    //    temp_kernel_value++;
    //}
    m_kenerl_number = inputparam.convolution_param().kernel_param().shape().dim(0);

    m_kenerl_channels = inputparam.convolution_param().kernel_param().shape().dim(1);

    if (0 != this->bottom_data_size[0].data_dim[1] % m_kenerl_channels)
    {
        return -1;
    }

    m_group_ = inputparam.convolution_param().group();
    m_stride_h = inputparam.convolution_param().stride_height();
    m_stride_w = inputparam.convolution_param().stride_width();
    m_pad_h = inputparam.convolution_param().pad_height();
    m_pad_w = inputparam.convolution_param().pad_width();
    m_dilation_h = inputparam.convolution_param().dilation_height();
    m_dilation_w = inputparam.convolution_param().dilation_width();
    if (inputparam.convolution_param().has_bias_param())
    {
        int temp_biasnum = inputparam.convolution_param().bias_param().data_size();

        for (int i = 0; i <temp_biasnum; i++)
        {
			float temp_biasvalue = inputparam.convolution_param().bias_param().data(i);
			if (temp_biasvalue < FLT_EPSILON && -temp_biasvalue < FLT_EPSILON) temp_biasvalue = 0;
            m_bias_value.push_back(temp_biasvalue);
        }
    }
    m_kernel_h = inputparam.convolution_param().kernel_height();
    m_kernel_w = inputparam.convolution_param().kernel_height();

	bool is_1x1_conv = m_kernel_h == 1 && m_kernel_w == 1 && m_pad_h == 0 && m_pad_w == 0 && m_stride_h == 1 && m_stride_w == 1;

	if (inputparam.convolution_param().has_tf_padding())
	{
		m_tf_padding = inputparam.convolution_param().tf_padding();
	}

    int output_h;
    int output_w;
    Caculate(this->bottom_data_size[0].data_dim[2], this->bottom_data_size[0].data_dim[3], m_kernel_h, m_kernel_w, m_pad_h, m_pad_w, m_stride_h, m_stride_w, m_dilation_h, m_dilation_w, output_h, output_w);


    this->top_data_size.resize(1);
    this->top_data_size[0].data_dim.resize(4);
    this->top_data_size[0].data_dim[2] = output_h;
    this->top_data_size[0].data_dim[3] = output_w;
    this->top_data_size[0].data_dim[1] = m_kenerl_number;
    this->top_data_size[0].data_dim[0] = this->bottom_data_size[0].data_dim[0];

    int length = output_h*output_w*m_group_;

    kernel_dims_ = m_kernel_h*m_kernel_w*m_kenerl_channels;
    col_buffer_shape_.push_back(kernel_dims_*m_group_);
    col_buffer_shape_.push_back(output_h);
    col_buffer_shape_.push_back(output_w);

    int64_t memory_size = m_kernel_h*m_kernel_w*m_kenerl_channels*m_group_*(output_h*output_w);
   
    //UpdateNetResourceMemory(m_p_holiday_net_resource, col_buffer_shape_);
	if (!is_1x1_conv) {
		m_p_holiday_net_resource->UpdateNetResourceMemory(col_buffer_shape_);
	}
    conv_out_spatial_dim_ = output_h* output_w;
    col_offset_ = kernel_dims_ * conv_out_spatial_dim_;
    weight_offset_ = m_kenerl_number * kernel_dims_ / m_group_;
    output_offset_ = this->top_data_size[0].data_dim[1] * conv_out_spatial_dim_ / m_group_;

    

    return 0;
}

template<class T>
int HolidayConvolutionCPU<T>::Caculate(const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int dilation_h, const int dilation_w,
    int& output_h, int& output_w)
{
	if (m_tf_padding == "VALID")
	{
		output_h = ceil((height + 2 * pad_h -
			(dilation_h * (kernel_h - 1))) / float(stride_h));
		output_w = ceil((width + 2 * pad_w -
			(dilation_w * (kernel_w - 1))) / float(stride_w));
	}
	else if (m_tf_padding == "SAME")
	{
		output_h = ceil((height + 2 * pad_h) / float(stride_h));
		output_w = ceil((width + 2 * pad_w) / float(stride_w));

		int original_view_h = height + 2 * pad_h;
		int original_view_w = width + 2 * pad_w;

		int caffe_need_view_h = output_h * stride_h + kernel_h - 1;
        int caffe_need_view_w = output_w * stride_w + kernel_w - 1;

        m_tf_fake_padding_h = (caffe_need_view_h - original_view_h) / 2;
        m_tf_fake_padding_w = (caffe_need_view_w - original_view_w) / 2;

        int tf_need_view_h = (output_h - 1) * stride_h + kernel_h;
        int tf_need_view_w = (output_w - 1) * stride_w + kernel_w;

        m_tf_conv_shift_h = -m_tf_fake_padding_h + (tf_need_view_h - original_view_h) / 2;
        m_tf_conv_shift_w = -m_tf_fake_padding_w + (tf_need_view_w - original_view_w) / 2;
	}
	else
	{
		output_h = (height + 2 * pad_h -
			(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
		output_w = (width + 2 * pad_w -
			(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	}

    return 0;
}

template<class T>
int HolidayConvolutionCPU<T>::Process(std::vector<HolidayFeatureMap<T>*> input_data_map, std::vector<HolidayFeatureMap<T>*>&output_data_map)
{
	T* output = output_data_map[0]->m_cpu.data();
	T*  input = input_data_map[0]->m_cpu.data();
	int num = input_data_map[0]->data_shape[0];
    output_data_map[0]->dwStorageType = DATA_CPU_WIDTH;

    //output_data_map[0]->m_cpu.shape_[0] = input_data_map[0]->m_cpu.shape_[0];
    output_data_map[0]->data_shape[0] = input_data_map[0]->data_shape[0];

    Caculate(input_data_map[0]->data_shape[2], input_data_map[0]->data_shape[3], m_kernel_h, m_kernel_w, m_pad_h, m_pad_w,
        m_stride_h, m_stride_w, m_dilation_h, m_dilation_w, output_data_map[0]->data_shape[2], output_data_map[0]->data_shape[3]);

    output_data_map[0]->data_shape[1] = m_kenerl_number;

    //output_data_map[0]->data_shape[1] = this->top_data_size[0].data_dim[1];
    //output_data_map[0]->data_shape[2] = this->top_data_size[0].data_dim[2];
    //output_data_map[0]->data_shape[3] = this->top_data_size[0].data_dim[3];

    conv_out_spatial_dim_ = output_data_map[0]->data_shape[2] * output_data_map[0]->data_shape[3];
    col_offset_ = kernel_dims_ * conv_out_spatial_dim_;

    int output_number_offset = output_data_map[0]->data_shape[1] * output_data_map[0]->data_shape[2] * output_data_map[0]->data_shape[3];;
    int input_number_offset = input_data_map[0]->data_shape[1] * input_data_map[0]->data_shape[2] * input_data_map[0]->data_shape[3];

    T* weights = m_p_kernel_blob->dataMemoryPtr();
    //T* weights_out = weights;

	bool fuse_bias = !m_bias_value.empty();
	if (fuse_bias)
	{
		SetBiasBlob(output_data_map[0]->m_cpu, output_data_map[0]->data_shape, m_bias_value);
	}

	bool is_1x1_conv = m_kernel_h == 1 && m_kernel_w == 1 && m_pad_h == 0 && m_pad_w == 0 && m_stride_h == 1 && m_stride_w == 1;

    int multi_number = m_kenerl_number / m_group_;
    for (int n = 0; n < num; ++n)
    {
		T* col_buff = nullptr;

		if (is_1x1_conv)
		{
			col_buff = input;
		}
		else
		{
			col_buff = m_p_holiday_net_resource->col_buffer_.data();
			conv_im2col_cpu(input_data_map[0], input, col_buff);
		}


        /*T* col_buff1 = col_buff;
        std::fstream fstmp("D:/Caffe/models_out/colbuf.txt", std::ios::out);
        for (int i = 0; i < 1098075; i++)
        {
        fstmp << *col_buff1 << "\t";
        col_buff1++;
        }
        fstmp.close();*/

        for (int g = 0; g < m_group_; g++)
        {
//#ifdef USING_SIMOD
//            if (multi_number*conv_out_spatial_dim_*kernel_dims_<(1037232 + 1))
//            {
//                //caffe_cpu_gemm_simod(false, false, multi_number, conv_out_spatial_dim_, kernel_dims_,
//                //weights + weight_offset_ * g, col_buff + col_offset_ * g, output + output_offset_ * g);
//
//                gemm_cpu(false, false, multi_number, conv_out_spatial_dim_, kernel_dims_, (T)1.,
//                    weights + weight_offset_ * g, col_buff + col_offset_ * g, (T)0., output + output_offset_ * g);
//
//            }
//            else
//            {
//                caffe_cpu_gemm<T>(CblasNoTrans, CblasNoTrans, multi_number, conv_out_spatial_dim_, kernel_dims_,
//                    (T)1., weights + weight_offset_ * g, col_buff + col_offset_ * g, (T)0., output + output_offset_ * g);
//            }
//#else
//#endif // USING_SIMOD
            caffe_cpu_gemm<T>(CblasNoTrans, CblasNoTrans, multi_number, conv_out_spatial_dim_, kernel_dims_,
				(T)1., weights + weight_offset_ * g, col_buff + col_offset_ * g, fuse_bias ? (T)1. : (T)0., output + output_offset_ * g);


        }
        output += output_number_offset;
        input += input_number_offset;

    }
    // if (!m_bias_value.empty())
    // {
    //     //AddBiasBlob(output_data_map[0]->m_cpu, m_bias_value);
    //     AddBiasBlob(output_data_map[0]->m_cpu, output_data_map[0]->data_shape, m_bias_value);
    // }


    return 0;
}

template<class T>
HolidayConvolutionCPU<T>::~HolidayConvolutionCPU()
{
    m_p_holiday_net_resource = nullptr;
}
template<class T>
HolidayConvolutionCPU<T>::HolidayConvolutionCPU()
{
    m_p_holiday_net_resource = nullptr;
}



#endif