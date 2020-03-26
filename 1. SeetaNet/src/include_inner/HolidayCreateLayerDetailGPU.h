#ifndef _CREATE_LAYER_DETAIL_GPU_H__
#define _CREATE_LAYER_DETAIL_GPU_H__

#include "HolidayBaseLayer.h"
#include <map>


#include "HolidayLayerType.h"
#include "HolidayCNN_proto.pb.h"
#include "HolidayNetResource.h"

#include "HolidayMemoryDataLayerGPU.h"
#include "HolidayConvolutionGPU.h"
#include "HolidayLocalResponseNormalizeGPU.h"
#include "HolidayReluGPU.h"
#include "HolidayPoolingGPU.h"
#include "HolidayInnerproductGPU.h"
#include "HolidaySoftmaxGPU.h"
#include "HolidayEltwiseGPU.h"
#include "HolidayConcatGPU.h"
#include "HolidayExpGPU.h"
#include "HolidayPowerGPU.h"
//#include "HolidaySliceGPU.h"
#include "HolidayBatchNormalizeGPU.h"
#include "HolidayScaleGPU.h"
#include "HolidaySplitGPU.h"
#include "HolidayPreReluGPU.h"
#include "HolidaySigmoidGPU.h"
#include "HolidayDeconvolutionGPU.h"
#include "HolidaySpaceToBatchNDGPU.h"
#include "HolidayBatchToSpaceNDGPU.h"
#include "HolidayReshapeGPU.h"
#include "HolidayRealMulGPU.h"

template <class	DataType>
int CreateMemoryDataFunctionGPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource);
template <class	DataType>
int CreateConvolutionFunctionGPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource);
template <class	DataType>
int CreateLRNFunctionGPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource);
template <class	DataType>
int CreateReluFunctionGPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource);
template <class	DataType>
int CreateReshapeFunctionGPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource);
template <class	DataType>
int CreatePoolingFunctionGPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource);
template <class	DataType>
int CreateInnerproductionFunctionGPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource);
template <class	DataType>
int CreateSoftmaxFunctionGPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource);
template <class	DataType>
int CreateEltwiseFunctionGPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource);
template <class	DataType>
int CreateConcatFunctionGPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource);
template <class	DataType>
int CreateExpFunctionGPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource);
template <class	DataType>
int CreatePowerFunctionGPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource);
template <class	DataType>
int CreateSliceFunctionGPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource);
template <class	DataType>
int CreateBatchNormliseFunctionGPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource);
template <class	DataType>
int CreateScaleFunctionGPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource);
template <class	DataType>
int CreateRealMulFunctionGPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource);
template <class	DataType>
int CreateSplitFunctionGPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource);
template <class	DataType>
int CreatePreReLUFunctionGPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource);
template <class	DataType>
int CreateSigmoidFunctionGPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource);
template <class	DataType>
int CreateSpaceToBatchNDFunctionGPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource);
template <class	DataType>
int CreateBatchToSpaceNDFunctionGPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource);


template <>
int CreateMemoryDataFunctionGPU(HolidayBaseLayer<float>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<float> *pNetResource)
{
	output_layer = new HolidayMemoryDataLayerGPU();
	output_layer->Init(layer_param, pNetResource);
	for (int i = 0; i < layer_param.bottom_index_size(); i++)
	{
		output_layer->bottom_index.push_back(layer_param.bottom_index(i));
	}
	for (int i = 0; i < layer_param.top_index().size(); i++)
	{
		output_layer->top_index.push_back(layer_param.top_index(i));
	}



	return 0;
}


template <>
int CreateConvolutionFunctionGPU(HolidayBaseLayer<float>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<float> *pNetResource)
{
	
	output_layer = new HolidayConvolutionGPU();
	output_layer->Init(layer_param, pNetResource);
	for (int i = 0; i < layer_param.bottom_index_size(); i++)
	{
		output_layer->bottom_index.push_back(layer_param.bottom_index(i));
	}
	for (int i = 0; i < layer_param.top_index().size(); i++)
	{
		output_layer->top_index.push_back(layer_param.top_index(i));
	}

	return 0;
}

template <>
int CreateLRNFunctionGPU(HolidayBaseLayer<float>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<float> *pNetResource)
{
	output_layer = new HolidayLocalResponseNormalizeGPU();
	output_layer->Init(layer_param, pNetResource);
	for (int i = 0; i < layer_param.bottom_index_size(); i++)
	{
		output_layer->bottom_index.push_back(layer_param.bottom_index(i));
	}
	for (int i = 0; i < layer_param.top_index().size(); i++)
	{
		output_layer->top_index.push_back(layer_param.top_index(i));
	}

	return 0;
}

template <>
int CreateReluFunctionGPU(HolidayBaseLayer<float>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<float> *pNetResource)
{
	output_layer = new HolidayReluGPU();
	output_layer->Init(layer_param, pNetResource);
	for (int i = 0; i < layer_param.bottom_index_size(); i++)
	{
		output_layer->bottom_index.push_back(layer_param.bottom_index(i));
	}
	for (int i = 0; i < layer_param.top_index().size(); i++)
	{
		output_layer->top_index.push_back(layer_param.top_index(i));
	}

	return 0;
}

template <>
int CreateReshapeFunctionGPU(HolidayBaseLayer<float>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<float> *pNetResource)
{
	output_layer = new HolidayReshapeGPU();
	output_layer->Init(layer_param, pNetResource);
	for (int i = 0; i < layer_param.bottom_index_size(); i++)
	{
		output_layer->bottom_index.push_back(layer_param.bottom_index(i));
	}
	for (int i = 0; i < layer_param.top_index().size(); i++)
	{
		output_layer->top_index.push_back(layer_param.top_index(i));
	}

	return 0;
}

template <>
int CreatePoolingFunctionGPU(HolidayBaseLayer<float>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<float> *pNetResource)
{
	output_layer = new HolidayPoolingGPU();
	output_layer->Init(layer_param, pNetResource);
	for (int i = 0; i < layer_param.bottom_index_size(); i++)
	{
		output_layer->bottom_index.push_back(layer_param.bottom_index(i));
	}
	for (int i = 0; i < layer_param.top_index().size(); i++)
	{
		output_layer->top_index.push_back(layer_param.top_index(i));
	}

	return 0;
}

template <>
int CreateInnerproductionFunctionGPU(HolidayBaseLayer<float>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<float> *pNetResource)
{
	output_layer = new HolidayInnerProductGPU();
	output_layer->Init(layer_param, pNetResource);
	for (int i = 0; i < layer_param.bottom_index_size(); i++)
	{
		output_layer->bottom_index.push_back(layer_param.bottom_index(i));
	}
	for (int i = 0; i < layer_param.top_index().size(); i++)
	{
		output_layer->top_index.push_back(layer_param.top_index(i));
	}

	return 0;
}

template <>
int CreateSoftmaxFunctionGPU(HolidayBaseLayer<float>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<float> *pNetResource)
{
	output_layer = new HolidaySoftMaxGPU();
	output_layer->Init(layer_param, pNetResource);
	for (int i = 0; i < layer_param.bottom_index_size(); i++)
	{
		output_layer->bottom_index.push_back(layer_param.bottom_index(i));
	}
	for (int i = 0; i < layer_param.top_index().size(); i++)
	{
		output_layer->top_index.push_back(layer_param.top_index(i));
	}

	return 0;
}

template <>
int CreateEltwiseFunctionGPU(HolidayBaseLayer<float>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<float> *pNetResource)
{
	output_layer = new HolidayEltwiseGPU();
	output_layer->Init(layer_param, pNetResource);
	for (int i = 0; i < layer_param.bottom_index_size(); i++)
	{
		output_layer->bottom_index.push_back(layer_param.bottom_index(i));
	}
	for (int i = 0; i < layer_param.top_index().size(); i++)
	{
		output_layer->top_index.push_back(layer_param.top_index(i));
	}

	return 0;
}
template <>
int CreateConcatFunctionGPU(HolidayBaseLayer<float>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<float> *pNetResource)
{
	output_layer = new HolidayConcatGPU();
	output_layer->Init(layer_param, pNetResource);
	for (int i = 0; i < layer_param.bottom_index_size(); i++)
	{
		output_layer->bottom_index.push_back(layer_param.bottom_index(i));
	}
	for (int i = 0; i < layer_param.top_index().size(); i++)
	{
		output_layer->top_index.push_back(layer_param.top_index(i));
	}

	return 0;
}
template <>
int CreateExpFunctionGPU(HolidayBaseLayer<float>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<float> *pNetResource)
{
	output_layer = new HolidayExpGPU();
	output_layer->Init(layer_param, pNetResource);
	for (int i = 0; i < layer_param.bottom_index_size(); i++)
	{
		output_layer->bottom_index.push_back(layer_param.bottom_index(i));
	}
	for (int i = 0; i < layer_param.top_index().size(); i++)
	{
		output_layer->top_index.push_back(layer_param.top_index(i));
	}

	return 0;
}
template <>
int CreatePowerFunctionGPU(HolidayBaseLayer<float>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<float> *pNetResource)
{
	output_layer = new HolidayPowerGPU();
	output_layer->Init(layer_param, pNetResource);
	for (int i = 0; i < layer_param.bottom_index_size(); i++)
	{
		output_layer->bottom_index.push_back(layer_param.bottom_index(i));
	}
	for (int i = 0; i < layer_param.top_index().size(); i++)
	{
		output_layer->top_index.push_back(layer_param.top_index(i));
	}

	return 0;
}
//template <>
//int CreateSliceFunctionGPU(HolidayBaseLayer<float>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource *pNetResource)
//{
//	output_layer = new HolidaySliceGPU();
//	output_layer->Init(layer_param, pNetResource);
//	for (int i = 0; i < layer_param.bottom_index_size(); i++)
//	{
//		output_layer->bottom_index.push_back(layer_param.bottom_index(i));
//	}
//	for (int i = 0; i < layer_param.top_index().size(); i++)
//	{
//		output_layer->top_index.push_back(layer_param.top_index(i));
//	}
//
//	return 0;
//}
template <>
int CreateBatchNormliseFunctionGPU(HolidayBaseLayer<float>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<float> *pNetResource)
{
	output_layer = new HolidayBatchNormalizeGPU();
	output_layer->Init(layer_param, pNetResource);
	for (int i = 0; i < layer_param.bottom_index_size(); i++)
	{
		output_layer->bottom_index.push_back(layer_param.bottom_index(i));
	}
	for (int i = 0; i < layer_param.top_index().size(); i++)
	{
		output_layer->top_index.push_back(layer_param.top_index(i));
	}

	return 0;
}
template <>
int CreateScaleFunctionGPU(HolidayBaseLayer<float>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<float> *pNetResource)
{
	output_layer = new HolidayScaleGPU();
	output_layer->Init(layer_param, pNetResource);
	for (int i = 0; i < layer_param.bottom_index_size(); i++)
	{
		output_layer->bottom_index.push_back(layer_param.bottom_index(i));
	}
	for (int i = 0; i < layer_param.top_index().size(); i++)
	{
		output_layer->top_index.push_back(layer_param.top_index(i));
	}

	return 0;
}

template <>
int CreateRealMulFunctionGPU(HolidayBaseLayer<float>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<float> *pNetResource)
{
	output_layer = new HolidayRealMulGPU();
	output_layer->Init(layer_param, pNetResource);
	for (int i = 0; i < layer_param.bottom_index_size(); i++)
	{
		output_layer->bottom_index.push_back(layer_param.bottom_index(i));
	}
	for (int i = 0; i < layer_param.top_index().size(); i++)
	{
		output_layer->top_index.push_back(layer_param.top_index(i));
	}

	return 0;
}

template <>
int CreateSplitFunctionGPU(HolidayBaseLayer<float>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<float> *pNetResource)
{
	output_layer = new HolidaySplitGPU();
	output_layer->Init(layer_param, pNetResource);
	for (int i = 0; i < layer_param.bottom_index_size(); i++)
	{
		output_layer->bottom_index.push_back(layer_param.bottom_index(i));
	}
	for (int i = 0; i < layer_param.top_index().size(); i++)
	{
		output_layer->top_index.push_back(layer_param.top_index(i));
	}

	return 0;
}
template <>
int CreatePreReLUFunctionGPU(HolidayBaseLayer<float>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<float> *pNetResource)
{
	output_layer = new HolidayPreReluGPU();
	output_layer->Init(layer_param, pNetResource);
	for (int i = 0; i < layer_param.bottom_index_size(); i++)
	{
		output_layer->bottom_index.push_back(layer_param.bottom_index(i));
	}
	for (int i = 0; i < layer_param.top_index().size(); i++)
	{
		output_layer->top_index.push_back(layer_param.top_index(i));
	}

	return 0;
}
template <>
int CreateSigmoidFunctionGPU(HolidayBaseLayer<float>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<float> *pNetResource)
{
	output_layer = new HolidaySigmoidGPU();
	output_layer->Init(layer_param, pNetResource);
	for (int i = 0; i < layer_param.bottom_index_size(); i++)
	{
		output_layer->bottom_index.push_back(layer_param.bottom_index(i));
	}
	for (int i = 0; i < layer_param.top_index().size(); i++)
	{
		output_layer->top_index.push_back(layer_param.top_index(i));
	}

	return 0;
}

template <class	DataType>
int CreateDeconvolutionFunctionGPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<float> *pNetResource)
{
    output_layer = new HolidayDeconvolutionGPU();
    output_layer->Init(layer_param, pNetResource);
    for (int i = 0; i < layer_param.bottom_index_size(); i++)
    {
        output_layer->bottom_index.push_back(layer_param.bottom_index(i));
    }
    for (int i = 0; i < layer_param.top_index().size(); i++)
    {
        output_layer->top_index.push_back(layer_param.top_index(i));
    }
    return 0;
}

template <>
int CreateSpaceToBatchNDFunctionGPU(HolidayBaseLayer<float>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<float> *pNetResource)
{
	output_layer = new HolidaySpaceToBatchNDGPU();
	output_layer->Init(layer_param, pNetResource);
	for (int i = 0; i < layer_param.bottom_index_size(); i++)
	{
		output_layer->bottom_index.push_back(layer_param.bottom_index(i));
	}
	for (int i = 0; i < layer_param.top_index().size(); i++)
	{
		output_layer->top_index.push_back(layer_param.top_index(i));
	}

	return 0;
}

template <>
int CreateBatchToSpaceNDFunctionGPU(HolidayBaseLayer<float>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<float> *pNetResource)
{
	output_layer = new HolidayBatchToSpaceNDGPU();
	output_layer->Init(layer_param, pNetResource);
	for (int i = 0; i < layer_param.bottom_index_size(); i++)
	{
		output_layer->bottom_index.push_back(layer_param.bottom_index(i));
	}
	for (int i = 0; i < layer_param.top_index().size(); i++)
	{
		output_layer->top_index.push_back(layer_param.top_index(i));
	}

	return 0;
}


#endif
