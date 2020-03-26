#ifndef HOLIDAY_NET_RESOURCE_H__
#define HOLIDAY_NET_RESOURCE_H__

#include <vector>
#include <map>

#include "MacroHoliday.h"

#include "HolidayCommon.h"
#include "HolidayBlobCpu.h"

template<class T>
struct HolidayShareParam
{
    std::map<int, HolidayBlobCpu<T> > param_map;
    int m_refrence_counts = 0;
	int m_device; // type of SeetaCNN_DEVICE_TYPE

#ifdef HOLIDAY_GPU
	int m_gpu_device_id;
#endif

};

template<class T>
struct HolidayNetResource
{
	int max_batch_size;
	
    HolidayShareParam<T>* m_shared_param;

	std::map<std::string, int> blob_name_map;
	std::vector<int> layer_type_vector;
	
	std::vector<HolidayDataSize> feature_vector_size;

	/* saving resized input */
	int m_new_width = -1;
	int m_new_height = -1;

    HolidayBlobCpu<T> col_buffer_;
    std::vector<int> col_buffer_shape_;

	int process_device_type;//cpu 0 or gpu 1
	int process_max_batch_size;

	int current_process_size;

    int colbuffer_memory_size;

    int CaculateMemorySize(std::vector<int> shape_vector)
    {
        int counts = 0;
        if (!shape_vector.empty())
        {
            counts = 1;
            for (int i = 0; i < shape_vector.size(); i++)
            {
                counts *= shape_vector[i];
            }
        }

        return counts;
    };
    
    int UpdateNetResourceMemory(std::vector<int> shape_vector)
    {
        int new_memory_size = CaculateMemorySize(shape_vector);
        if (new_memory_size > colbuffer_memory_size)
        {
            col_buffer_shape_ = shape_vector;
            colbuffer_memory_size = new_memory_size;

            col_buffer_.Reshape(shape_vector);
        }


        return 0;
    };

#ifdef HOLIDAY_GPU
	void *pNetResourceGpu;
#endif
};





#endif