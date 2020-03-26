#ifndef _CREATE_LAYER_MAP_CPU_H__
#define _CREATE_LAYER_MAP_CPU_H__


#include <map>

#include "HolidayLayerType.h"
#include "HolidayCNN_proto.pb.h"
#include "HolidayCreateLayerDetailCPU.h"

#include <stdint.h>

template<typename DType>
class HolidayBaseLayer;

template<typename DType>
class CreateLayerMapCPU
{
public:
	typedef int CREATE_NET_PARSEFUNCTION(HolidayBaseLayer<DType>*&, Holiday_LayerParameter&, HolidayNetResource<DType>*);

	static int(*FindRunFunciton(int32_t layertype))(HolidayBaseLayer<DType>*&, Holiday_LayerParameter&, HolidayNetResource<DType> *)
	{
		
		auto iteraiton_current = m_parse_function_map.find(layertype);

		int(*pfun)(HolidayBaseLayer<DType>*&, Holiday_LayerParameter&, HolidayNetResource<DType> *)(nullptr);

		if (iteraiton_current != m_parse_function_map.end())
		{
			pfun = iteraiton_current->second;
			
		}

		return pfun;
	}
private:
	static std::map<int32_t, int(*)(HolidayBaseLayer<DType>*&, Holiday_LayerParameter&, HolidayNetResource<DType>*)> CreateFunctionMap()
	{

		std::map<int32_t, int(*)(HolidayBaseLayer<DType>*&, Holiday_LayerParameter&, HolidayNetResource<DType>*)> FunctionMap;
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_MemoryDataLayer, CreateMemoryDataFunctionCPU<DType>));
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_ConvolutionLayer, CreateConvolutionFunctionCPU<DType>));
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_LRNLayer, CreateLRNFunctionCPU<DType>));
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_ReLULayer, CreateReluFunctionCPU<DType>));
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_PoolingLayer, CreatePoolingFunctionCPU<DType>));
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_InnerProductLayer, CreateInnerproductionFunctionCPU<DType>));
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_SoftmaxLayer, CreateSoftmaxFunctionCPU<DType>));

		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_EltwiseLayer, CreateEltwiseFunctionCPU<DType>));
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_ConcatLayer, CreateConcatFunctionCPU<DType>));
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_ExpLayer, CreateExpFunctionCPU<DType>));
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_PowerLayer, CreatePowerFunctionCPU<DType>));
		//FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_SliceLayer, CreateSliceFunctionCPU<DType>));
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_BatchNormliseLayer, CreateBatchNormliseFunctionCPU<DType>));
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_ScaleLayer, CreateScaleFunctionCPU<DType>));
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_SplitLayer, CreateSplitFunctionCPU<DType>));
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_PreReLULayer, CreatePreReLUFunctionCPU<DType>));
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_DeconvolutionLayer, CreateDeconvolutionFunctionCPU<DType>));
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_CropLayer, CreateCropLayerFunctionCPU<DType>));
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_SigmoidLayer, CreateSigmoidFunctionCPU<DType>));

		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_SpaceToBatchNDLayer, CreateSpaceToBatchNDFunctionCPU<DType>));
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_BatchToSpaceNDLayer, CreateBatchToSpaceNDFunctionCPU<DType>));

		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_ReshapeLayer, CreateReshapeFunctionCPU<DType>));
        FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_RealMulLayer, CreateRealMulFunctionCPU<DType>));

		return FunctionMap;
	};
	static const std::map<int32_t, int(*)(HolidayBaseLayer<DType>*&, Holiday_LayerParameter&, HolidayNetResource<DType> *)> m_parse_function_map;
};

template<> const std::map<int32_t, int(*)(HolidayBaseLayer<float>*&, Holiday_LayerParameter&, HolidayNetResource<float>*)> CreateLayerMapCPU<float>::m_parse_function_map = CreateLayerMapCPU<float>::CreateFunctionMap();
template<> const std::map<int32_t, int(*)(HolidayBaseLayer<double>*&, Holiday_LayerParameter&, HolidayNetResource<double>*)> CreateLayerMapCPU<double>::m_parse_function_map = CreateLayerMapCPU<double>::CreateFunctionMap();


#endif