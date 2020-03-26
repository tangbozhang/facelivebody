#ifndef _CREATE_LAYER_MAP_GPU_H__
#define _CREATE_LAYER_MAP_GPU_H__


#include <map>

#include "HolidayLayerType.h"
#include "HolidayCNN_proto.pb.h"
#include "HolidayCreateLayerDetailGPU.h"

#include <stdint.h>

template<typename DType>
class HolidayBaseLayer;

template<typename DType>
class CreateLayerMapGPU
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
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_MemoryDataLayer, CreateMemoryDataFunctionGPU<DType>));
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_ConvolutionLayer, CreateConvolutionFunctionGPU<DType>));
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_LRNLayer, CreateLRNFunctionGPU<DType>));
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_ReLULayer, CreateReluFunctionGPU<DType>));
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_PoolingLayer, CreatePoolingFunctionGPU<DType>));
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_InnerProductLayer, CreateInnerproductionFunctionGPU<DType>));
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_SoftmaxLayer, CreateSoftmaxFunctionGPU<DType>));
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_EltwiseLayer, CreateEltwiseFunctionGPU<DType>));
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_ConcatLayer, CreateConcatFunctionGPU<DType>));
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_ExpLayer, CreateExpFunctionGPU<DType>));
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_PowerLayer, CreatePowerFunctionGPU<DType>));
		//FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_SliceLayer, CreateSliceFunctionGPU<DType>));
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_BatchNormliseLayer, CreateBatchNormliseFunctionGPU<DType>));
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_ScaleLayer, CreateScaleFunctionGPU<DType>));
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_SplitLayer, CreateSplitFunctionGPU<DType>));
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_PreReLULayer, CreatePreReLUFunctionGPU<DType>));
		FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_SigmoidLayer, CreateSigmoidFunctionGPU<DType>));
        FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_DeconvolutionLayer, CreateDeconvolutionFunctionGPU<DType>));
        FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_SpaceToBatchNDLayer, CreateSpaceToBatchNDFunctionGPU<DType>));
        FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_BatchToSpaceNDLayer, CreateBatchToSpaceNDFunctionGPU<DType>));

        FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_ReshapeLayer, CreateReshapeFunctionGPU<DType>));
        FunctionMap.insert(std::pair<int32_t, CREATE_NET_PARSEFUNCTION*>(holiday_caffe::Enum_RealMulLayer, CreateRealMulFunctionGPU<DType>));

        return FunctionMap;
	};
	static const std::map<int32_t, int(*)(HolidayBaseLayer<DType>*&, Holiday_LayerParameter&, HolidayNetResource<DType> *)> m_parse_function_map;
};

template<> const std::map<int32_t, int(*)(HolidayBaseLayer<float>*&, Holiday_LayerParameter&, HolidayNetResource<float>*)> CreateLayerMapGPU<float>::m_parse_function_map = CreateLayerMapGPU<float>::CreateFunctionMap();
//template<> const std::map<int32_t, int(*)(HolidayBaseLayer<double>*&, Holiday_LayerParameter&, HolidayNetResource*)> CreateLayerMapGPU<double>::m_parse_function_map = CreateLayerMapGPU<double>::CreateFunctionMap();


#endif