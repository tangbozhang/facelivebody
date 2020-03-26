#ifndef _CREATE_LAYER_DETAIL_H__
#define _CREATE_LAYER_DETAIL_H__

#include "HolidayBaseLayer.h"
#include <map>


#include "HolidayLayerType.h"
#include "HolidayCNN_proto.pb.h"
#include "HolidayNetResource.h"

#include "HolidayMemoryDataLayerCPU.h"
#include "HolidayConvolutionCPU.h"
#include "HolidayLocalResponseNormalizeCPU.h"
#include "HolidayReluCPU.h"
#include "HolidayPoolingCPU.h"
#include "HolidayInnerproductCPU.h"
#include "HolidaySoftmaxCPU.h"
#include "HolidayConcatCPU.h"
#include "HolidayEltwiseCPU.h"
#include "HolidayExpCPU.h"
#include "HolidayPowerCPU.h"
//#include "HolidaySliceCPU.h"
#include "HolidaySpaceToBatchNDCPU.h"
#include "HolidayBatchToSpaceNDCPU.h"
#include "HolidayBatchNormliseCPU.h"
#include "HolidayScaleCPU.h"
#include "HolidaySplitCPU.h"
#include "HolidayPReluCPU.h"
#include "HolidayDeconvolutionCPU.h"
#include "HolidayCropCPU.h"
#include "HolidaySigmoidCPU.h"
#include "HolidayReshapeCPU.h"
#include "HolidayRealMulCPU.h"

template <class	DataType>
int CreateMemoryDataFunctionCPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource)
{
	output_layer = new HolidayMemoryDataLayerCPU<DataType>();
	output_layer->Init(layer_param, pNetResource);
	for (int i = 0; i < layer_param.bottom_index_size();i++)
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
int CreateConvolutionFunctionCPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource)
{
	output_layer = new HolidayConvolutionCPU<DataType>();
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
int CreateLRNFunctionCPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource)
{
	output_layer = new HolidayLocalResponseNormalizeCpu<DataType>();
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
int CreateReluFunctionCPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource)
{
	output_layer = new HolidayReluCPU<DataType>();
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
int CreateReshapeFunctionCPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource)
{
    output_layer = new HolidayReshapeCPU<DataType>();
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
int CreatePoolingFunctionCPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource)
{
	output_layer = new HolidayPoolingCpu<DataType>();
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
int CreateInnerproductionFunctionCPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource)
{
	output_layer = new HolidayInnerProductCPU<DataType>();
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
int CreateSoftmaxFunctionCPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource)
{
	output_layer = new HolidaySoftMaxCPU<DataType>();
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
int CreateConcatFunctionCPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource)
{
	output_layer = new HolidayConcatCPU<DataType>();
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
int CreateExpFunctionCPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource)
{
	output_layer = new HolidayExpCPU<DataType>();
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
int CreatePowerFunctionCPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource)
{
	output_layer = new HolidayPowerCPU<DataType>();
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

//template <class	DataType>
//int CreateSliceFunctionCPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource *pNetResource)
//{
//	output_layer = new HolidaySliceCPU<DataType>();
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

template <class	DataType>
int CreateBatchNormliseFunctionCPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource)
{
	output_layer = new HolidayBatchNormalizeCPU<DataType>();
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
int CreateScaleFunctionCPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource)
{
	output_layer = new HolidayScaleCPU<DataType>();
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
int CreateRealMulFunctionCPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource)
{
    output_layer = new HolidayRealMulCPU<DataType>();
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
int CreateSplitFunctionCPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource)
{
	output_layer = new HolidaySplitCPU<DataType>();
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
int CreatePreReLUFunctionCPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource)
{
	output_layer = new HolidayPReluCPU<DataType>();
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
int CreateEltwiseFunctionCPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource)
{
	output_layer = new HolidayEltwiseCPU<DataType>();
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
int CreateDeconvolutionFunctionCPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource)
{
	output_layer = new HolidayDeconvolutionCPU<DataType>();
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
int CreateCropLayerFunctionCPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource)
{
	output_layer = new HolidayCropCPU<DataType>();
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
int CreateSigmoidFunctionCPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource)
{
	output_layer = new HolidaySigmoidCPU<DataType>();
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
int CreateSpaceToBatchNDFunctionCPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource)
{
	output_layer = new HolidaySpaceToBatchNDCPU<DataType>();
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
int CreateBatchToSpaceNDFunctionCPU(HolidayBaseLayer<DataType>*& output_layer, Holiday_LayerParameter& layer_param, HolidayNetResource<DataType> *pNetResource)
{
    output_layer = new HolidayBatchToSpaceNDCPU<DataType>();
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