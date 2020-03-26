#include "HolidayNet.h"
#include "MemoryModel.h"
#include <vector>
#include <algorithm>

#include "HolidayNetResource.h"
#include "HolidayBaseLayer.h"
#include "HolidayFeatureMap.h"

#include "HolidayCreateLayerMapCPU.h"
#ifdef HOLIDAY_GPU
#include "HolidayCreateLayerMapGPU.h"
#include "HolidayCommonCuda.h"
#endif


//#include <windows.h> 
#include <algorithm>

#include "orz/mem/vat.h"
#include "orz/sync/shotgun.h"
#include "orz/tools/ctxmgr_lite.h"

struct HolidayNet
{
	std::vector< HolidayBaseLayer<NetF>* > Layer_vector;
	HolidayNetResource<NetF>* tmp_NetResource = nullptr;
	std::vector<HolidayFeatureMap<NetF>*> feature_vector_cpu;
	HolidayFeatureMap<NetF> input_data_blob;
	std::map<std::string, float*> feature_value_map;
	std::map<std::string, size_t> feature_value_size_map;

	orz::Vat vat;
	std::vector<int> blob_bottom_refs;
	std::vector<int> blob_top_refs;
	std::vector<int> output_blob_indexs;
	std::vector<int> keep_blob_indexs;
	// std::vector<std::string> blob_need_keep;

	std::shared_ptr<orz::Shotgun> gun;

#ifdef HOLIDAY_GPU
	int gpu_device_id;
#endif
};

int CreateNet(void* model, int max_batch_size, SeetaCNN_DEVICE_TYPE process_device_type, void** output_net_out, int gpu_device_id)
{
	CreateNetSharedParam(model, max_batch_size, process_device_type, output_net_out, nullptr, gpu_device_id);
	return 0;

}

static void SetRunningDevice(HolidayNet *net)
{
#ifdef HOLIDAY_GPU
	if (net->tmp_NetResource->process_device_type == HOLIDAY_CNN_GPU_DEVICE)
	{
		cudaSetDevice(net->gpu_device_id);
	}
#endif
}

int CreateNetSharedParam(void* model, int max_batchsize, SeetaCNN_DEVICE_TYPE process_device_type, void** output_net_out, void** output_shared_param, int gpu_device_id)
{
	// TODO: ����������early return ����Դ�ͷ�
#ifdef HOLIDAY_GPU
	auto device_count = 0;
	cudaGetDeviceCount(&device_count);
	gpu_device_id = std::min<int>(gpu_device_id, device_count - 1);
#endif
	HolidayShareParam<NetF>* tmp_output_shared_param = nullptr;
	if (nullptr != output_shared_param && nullptr != (*output_shared_param))
	{
		tmp_output_shared_param = (HolidayShareParam<NetF>*)(*output_shared_param);
		if (tmp_output_shared_param->m_device != process_device_type)
		{
			*output_net_out = nullptr;
			return MISSMATCH_DEVICE_ID;
		}

#ifdef HOLIDAY_GPU
		if (gpu_device_id < 0)
		{
			gpu_device_id = tmp_output_shared_param->m_gpu_device_id;
		}
		else if (process_device_type == HOLIDAY_CNN_GPU_DEVICE && tmp_output_shared_param->m_gpu_device_id != gpu_device_id)
		{
			*output_net_out = nullptr;
			return MISSMATCH_DEVICE_ID;
		}
#endif

		tmp_output_shared_param->m_refrence_counts = tmp_output_shared_param->m_refrence_counts + 1;

	}
	else
	{
		tmp_output_shared_param = new HolidayShareParam<NetF>;
		tmp_output_shared_param->m_refrence_counts = 1;
		tmp_output_shared_param->m_device = process_device_type;
#ifdef HOLIDAY_GPU
		tmp_output_shared_param->m_gpu_device_id = std::max<int>(gpu_device_id, 0);
#endif
	}

	HolidayNet* output_net_start = new HolidayNet();
	HolidayNet& output_net = *output_net_start;
	output_net.tmp_NetResource = new HolidayNetResource<NetF>;
	output_net.tmp_NetResource->max_batch_size = max_batchsize;
	output_net.tmp_NetResource->process_device_type = process_device_type;
	output_net.tmp_NetResource->colbuffer_memory_size = 0;

	output_net.tmp_NetResource->m_shared_param = tmp_output_shared_param;

	// output_net.gun = std::make_shared<orz::Shotgun>(4);

#ifdef HOLIDAY_GPU
	output_net.gpu_device_id = std::max<int>(gpu_device_id, 0);
#endif
	SetRunningDevice(&output_net);

	//output_net.Layer_vector
	MemoryModel* ptmp_model = (MemoryModel*)model;
	ptmp_model->model_mtx.lock();

	output_net.tmp_NetResource->m_new_height = ptmp_model->m_new_height;
	output_net.tmp_NetResource->m_new_width = ptmp_model->m_new_width;

	Holiday_LayerParameter* first_data_param = ptmp_model->all_layer_params[0];

	int blob_length = ptmp_model->vector_blob_names.size();
	output_net.tmp_NetResource->feature_vector_size.resize(blob_length);
	output_net.tmp_NetResource->feature_vector_size[0].data_dim.resize(4);
	output_net.tmp_NetResource->feature_vector_size[0].data_dim[0] = max_batchsize;
	output_net.tmp_NetResource->feature_vector_size[0].data_dim[1] = first_data_param->memory_data_param().channels();

	int local_input_height = output_net.tmp_NetResource->m_new_height > 0
		? output_net.tmp_NetResource->m_new_height
		: int(first_data_param->memory_data_param().height());
	int local_input_width = output_net.tmp_NetResource->m_new_width > 0
		? output_net.tmp_NetResource->m_new_width
		: int(first_data_param->memory_data_param().width());

	output_net.tmp_NetResource->feature_vector_size[0].data_dim[2] = local_input_height;
	output_net.tmp_NetResource->feature_vector_size[0].data_dim[3] = local_input_width;

	output_net.blob_bottom_refs.resize(blob_length, 0);
	output_net.blob_top_refs.resize(blob_length, 0);

	std::vector<int> input_data_shape = output_net.tmp_NetResource->feature_vector_size[0].data_dim;
	output_net.input_data_blob.data_shape.resize(4);
#ifdef FREE_DATA
	output_net.input_data_blob.m_cpu.ReshapeJustShape(input_data_shape);
#else
	output_net.input_data_blob.m_cpu.Reshape(input_data_shape);
#endif

	int return_pfun3 = 0;

	int layer_length = ptmp_model->all_layer_params.size();
	//layer_length = 3;
	output_net.Layer_vector.resize(layer_length, nullptr);

	for (int i = 0; i < ptmp_model->vector_blob_names.size(); i++)
	{
		output_net.tmp_NetResource->blob_name_map[ptmp_model->vector_blob_names[i]] = i;
	}

	if (HOLIDAY_CNN_GPU_DEVICE == output_net.tmp_NetResource->process_device_type)
	{
#ifdef HOLIDAY_GPU
		Holiday_NetParam_gpu djNetParam;
		djNetParam.dwMaxBatchNum = max_batchsize;
		int dwError = HolidayNetGPUInit(&djNetParam, (void **)&(output_net.tmp_NetResource->pNetResourceGpu));

		if (dwError < 0)
		{
			std::cout << "NET_GPU_INIT *** " << dwError << " *** " << std::endl;
			ptmp_model->model_mtx.unlock();
			return dwError;
		}
		output_net.input_data_blob.m_gpu.shape_.resize(4);
		output_net.input_data_blob.m_gpu.shape_[2] = local_input_height;
		output_net.input_data_blob.m_gpu.shape_[3] = local_input_width;
		output_net.input_data_blob.m_gpu.shape_[1] = first_data_param->memory_data_param().channels();
		output_net.input_data_blob.m_gpu.shape_[0] = max_batchsize;
		output_net.input_data_blob.m_gpu.Gpu_init(output_net.tmp_NetResource->pNetResourceGpu);
#endif
	}

	for (int i = 0; i < layer_length; i++)
	{

		CreateLayerMapCPU<NetF>::CREATE_NET_PARSEFUNCTION* pfun = nullptr;
		int layer_type = ptmp_model->all_layer_params[i]->type();
		std::string layer_name = ptmp_model->all_layer_params[i]->name();

#ifdef _DEBUG
		std::cout << "LOG: Creating layer(" << layer_type << "): " << layer_name << std::endl;
#endif

		if (HOLIDAY_CNN_CPU_DEVICE == output_net.tmp_NetResource->process_device_type
			|| ptmp_model->all_layer_params[i]->type() == holiday_caffe::Enum_SoftmaxLayer
			|| ptmp_model->all_layer_params[i]->type() == holiday_caffe::Enum_MemoryDataLayer
			|| ptmp_model->all_layer_params[i]->type() == holiday_caffe::Enum_CropLayer)
		{
			pfun = CreateLayerMapCPU<NetF>::FindRunFunciton(layer_type);
		}
#ifdef HOLIDAY_GPU
		else if (HOLIDAY_CNN_GPU_DEVICE == output_net.tmp_NetResource->process_device_type)
		{
			pfun = CreateLayerMapGPU<NetF>::FindRunFunciton(layer_type);
		}
#endif

		if (!pfun)
		{
			std::cerr << "ERROR: Unidentified layer(" << layer_type << "): " << layer_name << std::endl;
			ptmp_model->model_mtx.unlock();
			CNNReleaseNet(reinterpret_cast<void **>(&output_net_start));
			return UNIDENTIFIED_LAYER;
		}

		HolidayBaseLayer<NetF>* tmp_layer = nullptr;
		pfun(tmp_layer, *(ptmp_model->all_layer_params[i]), output_net.tmp_NetResource);
		//tmp_layer->m_layer_index = i;
		tmp_layer->m_layer_type = layer_type;

		std::vector<HolidayDataSize> tmp_vector_size;
		tmp_layer->GetTopSize(tmp_vector_size);
		std::vector<int> bottom_index;
		std::vector<int> top_index;
		for (int j = 0; j < ptmp_model->all_layer_params[i]->bottom_index().size(); j++)
		{
			bottom_index.push_back(ptmp_model->all_layer_params[i]->bottom_index(j));
		}
		for (int j = 0; j < ptmp_model->all_layer_params[i]->top_index().size(); j++)
		{
			top_index.push_back(ptmp_model->all_layer_params[i]->top_index(j));
		}

		for (int j = 0; j < tmp_vector_size.size(); j++)
		{
			HolidayDataSize current_size = tmp_vector_size[j];
			int index_value = top_index[j];
			output_net.tmp_NetResource->feature_vector_size[index_value] = current_size;
		}

		ptmp_model->all_layer_params[i]->top_index();
		output_net.Layer_vector[i] = tmp_layer;
	}
	output_net.feature_vector_cpu.resize(ptmp_model->vector_blob_names.size(), nullptr);
	for (int i = 0; i < output_net.feature_vector_cpu.size(); i++)
	{
		HolidayFeatureMap<NetF> * tmp_feature_map = new HolidayFeatureMap<NetF>();
		tmp_feature_map->pNetResource = output_net.tmp_NetResource;
		output_net.feature_vector_cpu[i] = tmp_feature_map;

	}

	for (int i = 0; i < layer_length; i++)
	{
		if (HOLIDAY_CNN_CPU_DEVICE == output_net.tmp_NetResource->process_device_type
			|| ptmp_model->all_layer_params[i]->type() == holiday_caffe::Enum_SoftmaxLayer
			|| ptmp_model->all_layer_params[i]->type() == holiday_caffe::Enum_MemoryDataLayer
			|| ptmp_model->all_layer_params[i]->type() == holiday_caffe::Enum_CropLayer)
		{
			//init blob as output
			for (int j = 0; j < output_net.Layer_vector[i]->top_index.size(); j++)
			{
				int index_blob = output_net.Layer_vector[i]->top_index[j];
				std::vector<int> shape_vector;
				shape_vector = output_net.tmp_NetResource->feature_vector_size[index_blob].data_dim;
				//(*pNet)->feature_vector_cpu[index_blob]->m_data2.resize(maxBatchSize, rows, cols, slices);
#ifdef FREE_DATA
				output_net.feature_vector_cpu[index_blob]->m_cpu.ReshapeJustShape(shape_vector);
#else
				output_net.feature_vector_cpu[index_blob]->m_cpu.Reshape(shape_vector);
#endif
				// output_net.feature_vector_cpu[index_blob]->data_shape.resize(4);
				output_net.feature_vector_cpu[index_blob]->data_shape = shape_vector;

				output_net.blob_top_refs[index_blob]++;
			}
			//init blob as input
			for (int j = 0; j < output_net.Layer_vector[i]->bottom_index.size(); j++)
			{
				int index_blob = output_net.Layer_vector[i]->bottom_index[j];
				std::vector<int> shape_vector;
				shape_vector = output_net.tmp_NetResource->feature_vector_size[index_blob].data_dim;
#ifdef FREE_DATA
				output_net.feature_vector_cpu[index_blob]->m_cpu.ReshapeJustShape(shape_vector);
#else
				output_net.feature_vector_cpu[index_blob]->m_cpu.Reshape(shape_vector);
#endif
				// output_net.feature_vector_cpu[index_blob]->data_shape.resize(4);
				output_net.feature_vector_cpu[index_blob]->data_shape = shape_vector;

				output_net.blob_bottom_refs[index_blob]++;
			}
		}
		// @TODO: change GPU memory allocate
#ifdef HOLIDAY_GPU
		else if (HOLIDAY_CNN_GPU_DEVICE == output_net.tmp_NetResource->process_device_type)
		{
			//init blob as output
			for (int j = 0; j < output_net.Layer_vector[i]->top_index.size(); j++)
			{
				int index_blob = output_net.Layer_vector[i]->top_index[j];
				std::vector<int> shape_vector;
				shape_vector = output_net.tmp_NetResource->feature_vector_size[index_blob].data_dim;

				// output_net.feature_vector_cpu[index_blob]->data_shape.resize(4);
				output_net.feature_vector_cpu[index_blob]->data_shape = shape_vector;
				output_net.feature_vector_cpu[index_blob]->m_gpu.shape_ = shape_vector;
				output_net.feature_vector_cpu[index_blob]->m_gpu.n_max_num = max_batchsize;

				return_pfun3 = output_net.feature_vector_cpu[index_blob]->m_gpu.Gpu_init(output_net.tmp_NetResource->pNetResourceGpu);
				if (0 != return_pfun3)
				{
					std::cout << "Gpu_init" << return_pfun3 << "\n";
					break;
				}
			}
			if (0 != return_pfun3)
			{
				break;
			}
			//init blob as input
			for (int j = 0; j < output_net.Layer_vector[i]->bottom_index.size(); j++)
			{
				int index_blob = output_net.Layer_vector[i]->bottom_index[j];
				std::vector<int> shape_vector;
				shape_vector = output_net.tmp_NetResource->feature_vector_size[index_blob].data_dim;

				// output_net.feature_vector_cpu[index_blob]->data_shape.resize(4);
				output_net.feature_vector_cpu[index_blob]->data_shape = shape_vector;
				output_net.feature_vector_cpu[index_blob]->m_gpu.shape_ = shape_vector;
				output_net.feature_vector_cpu[index_blob]->m_gpu.n_max_num = max_batchsize;

				return_pfun3 = output_net.feature_vector_cpu[index_blob]->m_gpu.Gpu_init(output_net.tmp_NetResource->pNetResourceGpu);

				output_net.blob_bottom_refs[index_blob]++;
			}
			if (0 != return_pfun3)
			{
				std::cout << "Gpu_init" << return_pfun3 << "\n";
				break;
			}
		}
#endif
	}

	// mark output blob
	output_net.output_blob_indexs.clear();
	for (int i = 0; i < blob_length; ++i)
	{
		if (output_net.blob_top_refs[i] > output_net.blob_bottom_refs[i])
		{
			output_net.output_blob_indexs.emplace_back(i);
		}
	}

	if (output_shared_param)
	{
		*output_shared_param = tmp_output_shared_param;
	}

	*output_net_out = output_net_start;
	ptmp_model->model_mtx.unlock();
	return 0;
}

template<typename Dtype, typename Dtype_input>
void OpencvDataToBlob(Dtype_input*inputMat, int height, int width, int nchannels, int num, HolidayBlobCpu<Dtype>& output_cube)
{
	std::vector<int> shape_vector;
	shape_vector.push_back(num);
	shape_vector.push_back(nchannels);
	shape_vector.push_back(height);
	shape_vector.push_back(width);
	// @TODO: no need reshape here
#ifdef FREE_DATA
	output_cube.ReshapeJustShape(shape_vector);
#else
	output_cube.Reshape(shape_vector);
#endif

	std::vector<int> index_vector;
	index_vector.resize(4, 0);
	int index = 0;
	for (int n = 0; n < num; n++)
	{
		index_vector[0] = n;
		for (int i = 0; i < height; i++)
		{
			index_vector[2] = i;
			for (int j = 0; j < width; j++)
			{
				index_vector[3] = j;
				for (int nc = 0; nc < nchannels; nc++)
				{
					index_vector[1] = nc;
					Dtype_input value = inputMat[index++];
					output_cube.data_at(index_vector) = value;
				}
			}
		}
	}
}


template <typename Dtype_input, typename Dtype>
void InputData2Blob(SeetaCNN_InputOutputData *pinput_Data, HolidayBlobCpu<Dtype>& output_cube);

template <>
void InputData2Blob<char, float>(SeetaCNN_InputOutputData *pinput_Data, HolidayBlobCpu<float>& output_cube)
{
	OpencvDataToBlob<float, unsigned char>(reinterpret_cast<unsigned char *>(pinput_Data[0].data_point_char), pinput_Data[0].height, pinput_Data[0].width, pinput_Data[0].channel, pinput_Data[0].number, output_cube);
}

template <>
void InputData2Blob<float, float>(SeetaCNN_InputOutputData *pinput_Data, HolidayBlobCpu<float>& output_cube)
{
	OpencvDataToBlob<float, float>(pinput_Data[0].data_point_float, pinput_Data[0].height, pinput_Data[0].width, pinput_Data[0].channel, pinput_Data[0].number, output_cube);

}

template<typename Dtype, typename Dtype_input>
void OutWidthDataToBlob(Dtype_input* inputMat, int height, int width, int nchannels, int num, HolidayBlobCpu<Dtype>& output_blob)
{
	std::vector<int> shape_vector;
	shape_vector.push_back(num);
	shape_vector.push_back(nchannels);
	shape_vector.push_back(height);
	shape_vector.push_back(width);
	// @TODO: no need reshape here

#ifdef FREE_DATA
	output_blob.ReshapeJustShape(shape_vector);
#else
	output_blob.Reshape(shape_vector);
#endif
	//output_cube.resize(num, height, width, nchannels);

	std::vector<int> index_vector;
	index_vector.resize(4, 0);
	int index = 0;
	for (int n = 0; n < num; n++)
	{
		index_vector[0] = n;
		for (int nc = 0; nc < nchannels; nc++)
		{
			index_vector[1] = nc;
			for (int i = 0; i < height; i++)
			{
				index_vector[2] = i;
				for (int j = 0; j < width; j++)
				{
					index_vector[3] = j;
					output_blob.data_at(index_vector) = inputMat[index++];
				}
			}
		}
	}
}



template <typename Dtype>
int RunNetTemplate(HolidayNet* output_net, int counts, SeetaCNN_InputOutputData* pinput_Data, int input_type)
{
	SetRunningDevice(output_net);

    orz::ctx::lite::bind<orz::Shotgun> _bind_gun(output_net->gun.get());

#ifdef FREE_DATA
	// prepare input blob size, not change input blob for now
	output_net->input_data_blob.m_cpu.dispose();
	// dispose all blob, not input_blob
	for (auto &blob : output_net->feature_vector_cpu)
	{
		blob->m_cpu.dispose();
	}
#endif

	output_net->input_data_blob.data_shape[0] = pinput_Data[0].number;
	output_net->input_data_blob.data_shape[1] = pinput_Data[0].channel;
	output_net->input_data_blob.data_shape[2] = pinput_Data[0].height;
	output_net->input_data_blob.data_shape[3] = pinput_Data[0].width;

#ifdef FREE_DATA
	output_net->input_data_blob.m_cpu.ReshapeJustShape(output_net->input_data_blob.data_shape);
	auto &input_blob = output_net->input_data_blob;
	input_blob.m_cpu.set_raw_data(output_net->vat.calloc_shared<NetF>(input_blob.m_cpu.count()));
#else
	output_net->input_data_blob.m_cpu.Reshape(output_net->input_data_blob.data_shape);
#endif

	if ((pinput_Data[0].number < 0)
		|| (pinput_Data[0].number > output_net->tmp_NetResource->feature_vector_size[0].data_dim[0]))
	{
		return -1;
	}

	if (HOLIDAY_CNN_CPU_DEVICE == output_net->tmp_NetResource->process_device_type)
	{
		//InputData2Blob<Dtype, float>(pinput_Data, output_net->input_data_blob.m_cpu);
		// OpencvDataToBlob<float, Dtype>(pinput_Data[0].data_point_char, pinput_Data[0].height, pinput_Data[0].width, pinput_Data[0].channel, pinput_Data[0].number, output_net->input_data_blob.m_cpu);
		if (input_type == SEETACNN_BGR_IMGE_CHAR || input_type == SEETACNN_BGR_IMGE_FLOAT)
		{
			InputData2Blob<Dtype, float>(pinput_Data, output_net->input_data_blob.m_cpu);
			output_net->input_data_blob.dwStorageType = DATA_CPU_WIDTH;
		}
		else if (SEETACNN_NCHW_FLOAT == input_type)
		{
			OutWidthDataToBlob<float, float>(pinput_Data[0].data_point_float, pinput_Data[0].height, pinput_Data[0].width, pinput_Data[0].channel, pinput_Data[0].number, output_net->input_data_blob.m_cpu);
			output_net->input_data_blob.dwStorageType = DATA_CPU_WIDTH;
		}
		else
		{
			return -1;
		}
	}
	// @TODO: change memory method
#ifdef HOLIDAY_GPU
	else if (HOLIDAY_CNN_GPU_DEVICE == output_net->tmp_NetResource->process_device_type)
	{
		if (input_type == SEETACNN_BGR_IMGE_CHAR || input_type == SEETACNN_BGR_IMGE_FLOAT)
		{
			InputData2Blob<Dtype, float>(pinput_Data, output_net->input_data_blob.m_cpu);
			output_net->input_data_blob.dwStorageType = DATA_CPU_WIDTH;
		}
		else if (SEETACNN_NCHW_FLOAT == input_type)
		{
			OutWidthDataToBlob<float, float>(pinput_Data[0].data_point_float, pinput_Data[0].height, pinput_Data[0].width, pinput_Data[0].channel, pinput_Data[0].number, output_net->input_data_blob.m_cpu);
			output_net->input_data_blob.dwStorageType = DATA_CPU_WIDTH;
		}
		else
		{
			return -1;
		}

		output_net->input_data_blob.m_gpu.pfData_cpu = output_net->input_data_blob.cpu_ptr();
		output_net->input_data_blob.dwStorageType = DATA_CPU_WIDTH;
	}
#endif

	auto local_blob_bottom_refs = output_net->blob_bottom_refs;
	for (auto &blob_index : output_net->output_blob_indexs)
	{
		local_blob_bottom_refs[blob_index]++;
	}
	for (auto &blob_index : output_net->keep_blob_indexs)
	{
		local_blob_bottom_refs[blob_index]++;
	}

	int return_result = 0;
	std::vector<HolidayFeatureMap<NetF>*> tmp_data_vector;
	tmp_data_vector.push_back(&(output_net->input_data_blob));

	int run_length = output_net->Layer_vector.size();
	for (int i = 0; i < run_length; i++)
	{
		//std::cout << i << std::endl;
		std::vector<int64_t> tmp_bottom_index =
			output_net->Layer_vector[i]->bottom_index;
		std::vector<int64_t> tmp_top_index =
			output_net->Layer_vector[i]->top_index;

		std::vector<HolidayFeatureMap<NetF>*> bottom_blob_vector;
		std::vector<HolidayFeatureMap<NetF>*> top_blob_vector;

		for (int j = 0; j < tmp_bottom_index.size(); j++)
		{
			bottom_blob_vector.push_back(output_net->feature_vector_cpu[tmp_bottom_index[j]]);
		}

		for (int j = 0; j < tmp_top_index.size(); j++)
		{
#ifdef FREE_DATA
			// spiecl for inplace layer
			if (tmp_bottom_index.size() > j && tmp_bottom_index[j] == tmp_top_index[j])
			{

			}
			else
			{
				auto &blob = output_net->feature_vector_cpu[tmp_top_index[j]];
				blob->m_cpu.set_raw_data(output_net->vat.calloc_shared<NetF>(blob->m_cpu.count()));	// calloc top blob memory
			}
#endif
			top_blob_vector.push_back(output_net->feature_vector_cpu[tmp_top_index[j]]);
		}
		if (bottom_blob_vector.empty())
		{
			bottom_blob_vector = tmp_data_vector;
		}

		// Those log for debug prcess time
		// #define MYLOG
#ifdef MYLOG

		using namespace std::chrono;
		microseconds duration(0);
		auto start = system_clock::now();

#endif

		auto layer = output_net->Layer_vector[i];
		return_result = layer->Process(bottom_blob_vector, top_blob_vector);

#ifdef MYLOG

		auto end = system_clock::now();
		duration += duration_cast<microseconds>(end - start);

		double spent = 1.0 * duration.count() / 1000;

		std::vector<std::string> top_names;
		for (auto index : layer->top_index)
		{
			std::string blob_name = "Unkown";
			for (auto &bolb_name_index : output_net->tmp_NetResource->blob_name_map)
			{
				if (bolb_name_index.second == index)
				{
					blob_name = bolb_name_index.first;
					break;
				}
			}
			top_names.push_back(blob_name);
		}
		std::vector<std::string> bottom_names;
		for (auto index : layer->bottom_index)
		{
			std::string blob_name = "Unkown";
			for (auto &bolb_name_index : output_net->tmp_NetResource->blob_name_map)
			{
				if (bolb_name_index.second == index)
				{
					blob_name = bolb_name_index.first;
					break;
				}
			}
			bottom_names.push_back(blob_name);
		}


		std::cout << "Layer(" << i << "): " << spent << "ms" << std::endl;
		std::cout << "\t(";
		for (int t = 0; t < bottom_names.size(); ++t)
		{
			if (t) std::cout << ", ";
			std::cout << bottom_names[t];
		}
		std::cout << ") -> (";
		for (int t = 0; t < top_names.size(); ++t)
		{
			if (t) std::cout << ", ";
			std::cout << top_names[t];
		}
		std::cout << ")" << std::endl;
		std::cout << "\t(";
		for (int t = 0; t < bottom_blob_vector.size(); ++t)
		{
			if (t) std::cout << ", ";
			auto &blob = bottom_blob_vector[t];
			std::cout << "[" << blob->data_shape[0] << ", " << blob->data_shape[1] << ", " << blob->data_shape[2] << ", " << blob->data_shape[3] << "]";
		}
		std::cout << ") -> (";
		for (int t = 0; t < top_blob_vector.size(); ++t)
		{
			if (t) std::cout << ", ";
			auto &blob = top_blob_vector[t];
			std::cout << "[" << blob->data_shape[0] << ", " << blob->data_shape[1] << ", " << blob->data_shape[2] << ", " << blob->data_shape[3] << "]";
		}
		std::cout << ")" << std::endl;
#endif


#ifdef FREE_DATA
		// free input dummy blob
		input_blob.m_cpu.dispose();
		// free useless blob
		for (int j = 0; j < tmp_bottom_index.size(); j++)
		{
			auto blob_index = tmp_bottom_index[j];
			local_blob_bottom_refs[blob_index]--;
			if (local_blob_bottom_refs[blob_index] == 0)
			{
				output_net->feature_vector_cpu[blob_index]->m_cpu.dispose();
			}
		}
#endif

		//output_net.Layer_vector
		if (0 != return_result)
		{
			std::cout << "Layer(" << i << ")\t" << "error!" << return_result << std::endl;
			break;
		}
	}

	return return_result;
}

int RunNetChar(void* output_net, int counts, SeetaCNN_InputOutputData* pinput_Data)
{
	HolidayNet* tmp_output_net = (HolidayNet*)output_net;
	int return_result = RunNetTemplate<char>(tmp_output_net, counts, pinput_Data, pinput_Data[0].buffer_type);
	return return_result;
}

int RunNetFloat(void* output_net, int counts, SeetaCNN_InputOutputData* pinput_Data)
{
	HolidayNet* tmp_output_net = (HolidayNet*)output_net;
	int return_result = RunNetTemplate<float>(tmp_output_net, counts, pinput_Data, pinput_Data[0].buffer_type);
	return return_result;
}

int HolidayCNNGetFeatureMap(const char* buffer_name, void *pNetIn, SeetaCNN_InputOutputData* outputData)
{
	int index_value(0);
	HolidayNet* pNet = (HolidayNet*)pNetIn;
	SetRunningDevice(pNet);
	if (pNet->tmp_NetResource->blob_name_map.find(buffer_name) != pNet->tmp_NetResource->blob_name_map.end())
	{
		int index = pNet->tmp_NetResource->blob_name_map[buffer_name];

		outputData->number = pNet->feature_vector_cpu[index]->data_shape[0];
		outputData->buffer_type = SEETACNN_NCHW_FLOAT;
		outputData->channel = pNet->tmp_NetResource->feature_vector_size[index].data_dim[1];
		outputData->width = pNet->tmp_NetResource->feature_vector_size[index].data_dim[3];
		outputData->height = pNet->tmp_NetResource->feature_vector_size[index].data_dim[2];

		int size_memory = outputData->number * outputData->channel * outputData->height * outputData->width;

		std::vector<int32_t> out_feature_shape = pNet->feature_vector_cpu[index]->data_shape;
		outputData->number = out_feature_shape[0];
		outputData->channel = out_feature_shape[1];
		outputData->height = out_feature_shape[2];
		outputData->width = out_feature_shape[3];

		if (pNet->feature_value_map.find(buffer_name) == pNet->feature_value_map.end())
		{

			float* out_value_innerl(nullptr);
			out_value_innerl = new float[size_memory];
			memset(out_value_innerl, 0, size_memory * sizeof(float));
			pNet->feature_value_map.insert(std::pair<std::string, float*>(buffer_name, out_value_innerl));
			pNet->feature_value_size_map.insert(std::pair<std::string, size_t>(buffer_name, size_memory));
			outputData->data_point_float = out_value_innerl;
		}
		else
		{
			if (nullptr == pNet->feature_value_map[buffer_name])
			{
				pNet->feature_value_map[buffer_name] = new float[size_memory];
				pNet->feature_value_size_map[buffer_name] = static_cast<size_t>(size_memory);
			}
			else if (pNet->feature_value_size_map.find(buffer_name) == pNet->feature_value_size_map.end() ||
				pNet->feature_value_size_map[buffer_name] < size_memory) {
				delete[] pNet->feature_value_map[buffer_name];
				pNet->feature_value_map[buffer_name] = new float[size_memory];
				pNet->feature_value_size_map[buffer_name] = static_cast<size_t>(size_memory);
			}
			outputData->data_point_float = pNet->feature_value_map[buffer_name];
		}
		index_value = 0;

		if (pNet->feature_vector_cpu[index]->dwStorageType == DATA_CPU_WIDTH)
		{

			int64_t out_puts_counts = 1;

			for (int index_shape = 0; index_shape < out_feature_shape.size(); index_shape++)
			{
				out_puts_counts *= out_feature_shape[index_shape];
			}

			auto blob_data = pNet->feature_vector_cpu[index]->m_cpu.data();
			if (blob_data) {
				memcpy(outputData->data_point_float, blob_data, out_puts_counts * sizeof(float));
			}
		}

		// @TODO: change output GPU memory check (with memory manage)
#ifdef HOLIDAY_GPU
		else if (pNet->feature_vector_cpu[index]->dwStorageType == DATA_GPU)
		{
			pNet->feature_vector_cpu[index]->m_gpu.Gpu_DataOut(pNet->tmp_NetResource->pNetResourceGpu, DATA_CPU_WIDTH, outputData->data_point_float);
		}
#endif
		else {}




		return 0;
	}
	else
	{
		(outputData->data_point_float) = nullptr;
		return BLOB_NAME_NOT_EXIST;
	}
	return 0;
}

int HolidayCNNGetAllFeatureMap(void * pNetIn, int* number, SeetaCNN_InputOutputData** outputData)
{
	HolidayNet* pNet = (HolidayNet*)pNetIn;
	SetRunningDevice(pNet);
	int all_size = pNet->tmp_NetResource->blob_name_map.size();
	SeetaCNN_InputOutputData *outputDatatmp = new SeetaCNN_InputOutputData[all_size];
	*number = all_size;
	for (auto tmp_iter = pNet->tmp_NetResource->blob_name_map.begin(); tmp_iter != pNet->tmp_NetResource->blob_name_map.end(); tmp_iter++)
	{
		int index = pNet->tmp_NetResource->blob_name_map[tmp_iter->first];

		HolidayCNNGetFeatureMap(tmp_iter->first.c_str(), pNetIn, &(outputDatatmp[index]));
	}

	*outputData = outputDatatmp;


	return 0;
}

void HolidayCNNFreeAllFeatureMap(void * pNetIn, const SeetaCNN_InputOutputData *outputData)
{
	HolidayNet* pNet = (HolidayNet*)pNetIn;
	SetRunningDevice(pNet);
	delete[] outputData;
}


void CNNReleaseNet(void** pNetIn)
{
	if (*pNetIn)
	{
		HolidayNet* pNet = (HolidayNet*)*pNetIn;
		SetRunningDevice(pNet);

		for (auto tmp_iteration = pNet->feature_value_map.begin(); tmp_iteration != pNet->feature_value_map.end(); tmp_iteration++)
		{
			delete[] tmp_iteration->second;
			tmp_iteration->second = nullptr;
		}
		pNet->feature_value_map.clear();
		pNet->feature_value_size_map.clear();

		for (int i = 0; i < pNet->Layer_vector.size(); i++)
		{
			pNet->Layer_vector[i]->Exit();
			delete pNet->Layer_vector[i];
		}
		pNet->Layer_vector.clear();

		for (int i = 0; i < pNet->Layer_vector.size(); i++)
		{
			delete pNet->Layer_vector[i];
		}
		pNet->Layer_vector.clear();
		pNet->tmp_NetResource->blob_name_map.clear();

		for (int i = 0; i < pNet->feature_vector_cpu.size(); i++)
		{
#ifdef HOLIDAY_GPU
			pNet->feature_vector_cpu[i]->m_gpu.Gpu_free();
#endif
			delete pNet->feature_vector_cpu[i];
		}
		pNet->feature_vector_cpu.clear();

		if (HOLIDAY_CNN_GPU_DEVICE == pNet->tmp_NetResource->process_device_type)
		{
#ifdef HOLIDAY_GPU
			pNet->input_data_blob.m_gpu.Gpu_free();
			HolidayNetGPUExit((void *)pNet->tmp_NetResource->pNetResourceGpu);
#endif
		}
		else
		{
		}

		pNet->tmp_NetResource->m_shared_param->m_refrence_counts -= 1;

		if (0 == pNet->tmp_NetResource->m_shared_param->m_refrence_counts)
		{
			delete  pNet->tmp_NetResource->m_shared_param;
			pNet->tmp_NetResource->m_shared_param = nullptr;
		}

		if (pNet->tmp_NetResource)
		{
			delete pNet->tmp_NetResource;
			pNet->tmp_NetResource = nullptr;
		}

		// free the memory, controlled by pNet->vat, avoid memory leak
		pNet->input_data_blob.m_cpu.dispose();

		delete pNet;
		pNet = nullptr;
		*pNetIn = nullptr;
	}

}

int HolidayCNNReleaseSharedParam(void ** shared_param)
{

	*shared_param = nullptr;
	return 0;
}

void HolidayKeepBlob(struct SeetaCNN_Net* net, const char* blob_name)
{
	HolidayNet* inner_net = (HolidayNet*)net;
	SetRunningDevice(inner_net);

	auto it = inner_net->tmp_NetResource->blob_name_map.find(blob_name);
	if (it == inner_net->tmp_NetResource->blob_name_map.end()) return;

	inner_net->keep_blob_indexs.push_back(it->second);
}

void HolidayKeepNoBlob(struct SeetaCNN_Net* net)
{
	HolidayNet* inner_net = (HolidayNet*)net;
	SetRunningDevice(inner_net);
	inner_net->keep_blob_indexs.clear();
}

void HolidayKeepAllBlob(struct SeetaCNN_Net* net)
{
	HolidayNet* inner_net = (HolidayNet*)net;
	SetRunningDevice(inner_net);
	inner_net->keep_blob_indexs.clear();
	int blob_length = inner_net->feature_vector_cpu.size();
	for (int i = 0; i < blob_length; ++i) inner_net->keep_blob_indexs.push_back(i);
}

int HolidayHasKeptBlob(struct SeetaCNN_Net* net, const char* blob_name)
{
	HolidayNet* inner_net = (HolidayNet*)net;

	SetRunningDevice(inner_net);

	auto it = inner_net->tmp_NetResource->blob_name_map.find(blob_name);
	if (it == inner_net->tmp_NetResource->blob_name_map.end()) return 0;

	int blob_index = it->second;

#define HAS(vec, val) (std::find((vec).begin(), (vec).end(), (val)) != (vec).end())

	return HAS(inner_net->output_blob_indexs, blob_index) || HAS(inner_net->keep_blob_indexs, blob_index);

#undef HAS
}

void HolidaySetNumThreadsEx(struct SeetaCNN_Net* net, int num)
{
	HolidayNet* hnet = (HolidayNet*)net;
	hnet->gun = std::make_shared<orz::Shotgun>(num < 1 ? 0 : num);
}

void *GetNetSharedParam(void *net)
{
	HolidayNet* tmp_output_net = (HolidayNet*)net;
	SetRunningDevice(tmp_output_net);
	return tmp_output_net->tmp_NetResource->m_shared_param;
}
