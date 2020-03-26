#ifndef HOLIDAY_BASE_LAYER
#define HOLIDAY_BASE_LAYER

#include <stdint.h>
#include "HolidayCNN_proto.pb.h"
#include "HolidayCommon.h"
#include "HolidayFeatureMap.h"
#include "HolidayNetResource.h"


template<typename T>
class HolidayBaseLayer
{
public:
	HolidayBaseLayer(){};
	virtual ~HolidayBaseLayer() {};

	virtual int GetTopSize(std::vector<HolidayDataSize>& out_data_size);
	virtual int Exit(){ return 0; };
	virtual int Init(Holiday_LayerParameter& inputparam, HolidayNetResource<T> *pNetResource)
    {
        m_layer_index = inputparam.layer_index();
        return 0; 
    };
	virtual int Process(std::vector<HolidayFeatureMap<T>*> input_data_map, std::vector<HolidayFeatureMap<T>*>& output_data_map)
	{ return 0; };
public:
	std::vector<HolidayDataSize> bottom_data_size;
	std::vector<int64_t> bottom_index;

	std::vector<HolidayDataSize> top_data_size;
	std::vector<int64_t> top_index;
    int m_layer_index;
	int m_layer_type;
};

template <class	T>
int HolidayBaseLayer<T>::GetTopSize(std::vector<HolidayDataSize>& out_data_size)
{
	out_data_size = this->top_data_size;
	return 0;
}

#endif

