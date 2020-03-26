#include "VIPLFaceRecognizer.h"

#include <cmath>

#include <string>
#include <memory>

#include <HolidayForward.h>

#include "SeetaModelHeader.h"
#include "seeta/common_alignment.h"
#include <iostream>

#include "orz/io/jug/jug.h"
#include <functional>

#include "orz/utils/log.h"
#include "seeta/ImageProcess.h"

#include "orz/sync/shotgun.h"

#include <algorithm>
#include <orz/tools/ctxmgr_lite.h>

//#define USE_CBLAS
#define USE_SIMD
#ifdef ANDROID_PLATFORM
#undef USE_SIMD
#endif
#define USE_MULTI_THREADS

//#define DOUBLE_SQRT
//#define SINGLE_SQRT

class VIPLFaceRecognizer::Recognizer
{
public:
	// char *buffer = nullptr;	// remove this two for memory init
	// int64_t length;
	SeetaCNN_Model *model = nullptr;
	SeetaCNN_Net *net = nullptr;
	seeta::FRModelHeader header;

	Device device = AUTO;

	// for memory share
	SeetaCNN_SharedParam *param = nullptr;

	std::string version;
	std::string date;
	std::string name;
	std::function<float(float)> trans_func;

	static int max_batch_global;
	int max_batch_local;

	// for YuLe setting
    int sqrt_times = -1;
	std::string default_method = "crop";
    std::string method = "";
    int crop_width = 0;
    int crop_height = 0;

	// for multi thread setting
	static int core_number_global;
	int recognizer_number_threads = 2;
	std::shared_ptr<orz::Shotgun> gun;
	std::vector<SeetaCNN_Net *> cores;

    std::shared_ptr<orz::Shotgun> m_crop_gun;
    
    Recognizer()
    {
        header.width = 256;
        header.height = 256;
        header.channels = 3;
		max_batch_local = max_batch_global;
		recognizer_number_threads = core_number_global;
#ifdef USE_MULTI_THREADS
		gun = std::make_shared<orz::Shotgun>(recognizer_number_threads > 1 ? recognizer_number_threads : 0);
#endif
        m_crop_gun = std::make_shared<orz::Shotgun>(4);
    }

	void free()
	{
		// trans_func = nullptr;
		// if (buffer) SeetaFreeBuffer(buffer);
		// buffer = nullptr;
		if (model) SeetaReleaseModel(model);
		model = nullptr;
		if (net) SeetaReleaseNet(net);
		net = nullptr;
		for (size_t i = 1; i < cores.size(); ++i)
		{
			SeetaReleaseNet(cores[i]);
		}
		cores.clear();
	}

	float trans(float similar) const
	{
		if (trans_func)
		{
			return trans_func(similar);
		}
		return similar;
	}

	int GetMaxBatch()
    {
		return max_batch_local;
    }

	int GetCoreNumber()
	{
#ifdef USE_MULTI_THREADS
		return gun->size() > 0 ? int(gun->size()) : 1;
#else
		return 1;
#endif
    }

	~Recognizer()
	{
		Recognizer::free();
	}

	void fix() {
		if (this->sqrt_times < 0) {
			this->sqrt_times = this->header.feature_size >= 1024 ? 1 : 0;
		}

		if (this->method.empty()) {
			this->method = this->header.feature_size >= 1024 ? this->default_method : "resize";
		}
	}
};

int VIPLFaceRecognizer::Recognizer::max_batch_global = 1;
int VIPLFaceRecognizer::Recognizer::core_number_global = 1;

// static const SeetaCNN_InputOutputData seeta2holiday(const VIPLImageData &img)
// {
// 	SeetaCNN_InputOutputData himg;
// 	himg.number = 1;
// 	himg.channel = img.channels;
// 	himg.height = img.height;
// 	himg.width = img.width;
// 	himg.buffer_type = SEETACNN_BGR_IMGE_CHAR;
// 	himg.data_point_char = reinterpret_cast<unsigned char *>(img.data);
// 	return himg;
// }

float sigmoid(float x, float a = 0, float b = 1)
{
    return 1 / (1 + exp(a - b * x));
}

float poly(float x, const std::vector<float> &params)
{
    if (params.empty()) return x;
    float y = 0;

    for (size_t i = 0; i < params.size(); ++i)
    {
        int p = static_cast<int>(params.size() - 1 - i);
        y += params[i] * std::pow(x, p);
    }
    return std::max<float>(0, std::min<float>(1, y));
}

#if SEETA_LOCK_SDK
#include <lock/macro.h>
#endif
#include <orz/io/i.h>

VIPLFaceRecognizerModel::VIPLFaceRecognizerModel(const char* model_path, int device)
	: m_impl(new VIPLFaceRecognizer::Recognizer)
{
#ifdef SEETA_CHECK_INIT
	SEETA_CHECK_INIT;
#endif

	auto recognizer = reinterpret_cast<VIPLFaceRecognizer::Recognizer*>(m_impl);

	if (!model_path)
	{
		orz::Log(orz::FATAL) << "Can not load empty model" << orz::crash;
	}
	auto GPU0 = 3;
	// auto GPU7 = 10;
	int gpu_id = 0;
	SeetaCNN_DEVICE_TYPE type = HOLIDAY_CNN_CPU_DEVICE;
	switch (device)
	{
	case 0:	// AUTO
		type = SeetaDefaultDevice();
		break;
	case 1:	// CPU
		break;
	case 2:	// GPU
		if (SeetaDefaultDevice() == HOLIDAY_CNN_GPU_DEVICE) type = HOLIDAY_CNN_GPU_DEVICE;
		break;
	default: // GPU0~7
        if (device >= GPU0 && SeetaDefaultDevice() == HOLIDAY_CNN_GPU_DEVICE)
        {
            type = HOLIDAY_CNN_GPU_DEVICE;
            gpu_id = device - GPU0;
        }
	}

	std::string type_string;
	if (type == HOLIDAY_CNN_GPU_DEVICE) type_string = "GPU" + std::to_string(gpu_id);
	else type_string = "CPU";

	orz::Log(orz::INFO) << "RecognizerModel: " << "Using device " << type_string;
	recognizer->device = VIPLFaceRecognizer::Device(device);

	std::shared_ptr<char> sta_buffer;
	orz::binary sta_data;
	char *buffer = nullptr;
	int64_t buffer_len = 0;

	auto bin = orz::read_file(model_path);
	if (bin.empty())
	{
		orz::Log(orz::FATAL) << "Can not access \"" << model_path << "\"" << orz::crash;
	}
	
#ifdef SEETA_CHECK_LOAD
	SEETA_CHECK_LOAD(bin);
#endif

	orz::imemorystream infile(bin.data(), bin.size());

	int mark;
	orz::binio<int>::read(infile, mark);
	if (mark == 0x19910929)
	{
		auto model = orz::jug_read(infile);

#ifdef SEETA_CHECK_MODEL
		SEETA_CHECK_MODEL(model, "Using unauthorized model");
#endif

		recognizer->version = std::string(model["version"]);

		if (recognizer->version != "5.0")
		{
			throw orz::Exception("model version missmatch.");
		}

		auto single_data = model["data"];
		if (single_data.valid(orz::Piece::LIST)) single_data = single_data[0];

		auto single_name = model["name"];
		if (single_name.valid(orz::Piece::LIST)) single_name = single_name[0];

		sta_data = single_data.to_binary();
		recognizer->date = std::string(model["date"]);
		recognizer->name = std::string(single_name);
		std::string method = std::string(model["trans"]["method"]);


        if (method == "sigmoid")
        {
            float a = model["trans"]["params"][0];
            float b = model["trans"]["params"][1];
            recognizer->trans_func = std::bind(sigmoid, std::placeholders::_1, a, b);
        }
        else if (method == "poly")
        {
            auto params = model["trans"]["params"];
            std::vector<float> ps(params.size());
            for (size_t i = 0; i < params.size(); ++i) ps[i] = params[i];
            recognizer->trans_func = std::bind(poly, std::placeholders::_1, ps);
        }

        auto output = model["output"];
        if (output.valid()){
            recognizer->sqrt_times = orz::jug_get<int>(output["sqrt_times"], -1);
        }

		auto input = model["input"];
        if (input.valid()) {
            recognizer->method = orz::jug_get<std::string>(input["method"], "");
            recognizer->crop_height = orz::jug_get<int>(input["crop_height"], 0);
            recognizer->crop_width = orz::jug_get<int>(input["crop_width"], 0);
		}

		// return LoadModel(data.data(), data.size(), device);

		// this buffer for read only use
		buffer = sta_data.data<char>();
		buffer_len = sta_data.size();

	}
	else
	{
		infile.seekg(0, std::ios::beg);
		infile.seekg(0, std::ios::end);
		auto sta_length = infile.tellg();
		sta_buffer.reset(new char[size_t(sta_length)], std::default_delete<char[]>());
		infile.seekg(0, std::ios::beg);
		infile.read(sta_buffer.get(), sta_length);

		buffer = sta_buffer.get();
		buffer_len = sta_length;
	}

	// read header
	size_t header_size = recognizer->header.read(buffer, size_t(buffer_len));

	// convert the model
	if (SeetaReadModelFromBuffer(buffer + header_size, size_t(buffer_len - header_size), &recognizer->model))
	{
		orz::Log(orz::FATAL) << "Got an broken model file" << orz::crash;
	}
	// create the net
	int err_code;
	// notice: Model just batch 1
	if (type == HOLIDAY_CNN_GPU_DEVICE)
	{
		err_code = SeetaCreateNetGPUSharedParam(recognizer->model, 1, gpu_id, &recognizer->net, &recognizer->param);
	}
	else
	{
		err_code = SeetaCreateNetSharedParam(recognizer->model, 1, type, &recognizer->net, &recognizer->param);
	}

	if (err_code)
	{
		SeetaReleaseModel(recognizer->model);
		recognizer->model = nullptr;
		orz::Log(orz::FATAL) << "Can not init net from broken model" << orz::crash;
	}

	recognizer->fix();

	// here, we got model, net, and param
}

VIPLFaceRecognizerModel::~VIPLFaceRecognizerModel()
{
	auto recognizer = reinterpret_cast<VIPLFaceRecognizer::Recognizer*>(m_impl);
	delete recognizer;
}

const VIPLFaceRecognizer::Param* VIPLFaceRecognizer::GetParam() const
{
	return reinterpret_cast<const Param *>(SeetaGetSharedParam(recognizer->net));
}

VIPLFaceRecognizer::VIPLFaceRecognizer(const Param* param)
	: recognizer(new Recognizer)
{
#ifdef SEETA_CHECK_INIT
	SEETA_CHECK_INIT;
#endif
	recognizer->param =
		const_cast<SeetaCNN_SharedParam *>(
		reinterpret_cast<const SeetaCNN_SharedParam *>(
		param));
}

static std::vector<SeetaCNN_Net *> ExtendMultiNets(SeetaCNN_Net *pnet, int number, SeetaCNN_Model *pmodel, int max_batch, SeetaCNN_DEVICE_TYPE type, int id)
{
	auto shared_param = SeetaGetSharedParam(pnet);
	std::vector<SeetaCNN_Net *> cores(number);
	cores[0] = pnet;
	for (size_t i = 1; i < cores.size(); ++i)
	{
		if (type == HOLIDAY_CNN_GPU_DEVICE)
		{
			SeetaCreateNetGPUSharedParam(pmodel, max_batch, id, &cores[i], &shared_param);
		}
		else
		{
			SeetaCreateNetSharedParam(pmodel, max_batch, type, &cores[i], &shared_param);
		}
	}
	return std::move(cores);
}

VIPLFaceRecognizer::VIPLFaceRecognizer(const VIPLFaceRecognizerModel& model)
	: recognizer(new Recognizer)
{
#ifdef SEETA_CHECK_INIT
	SEETA_CHECK_INIT;
#endif
	auto other = reinterpret_cast<VIPLFaceRecognizer::Recognizer*>(model.m_impl);

	auto device = other->device;

    using self = VIPLFaceRecognizer;
	SeetaCNN_DEVICE_TYPE type = HOLIDAY_CNN_CPU_DEVICE;
	int gpu_id = 0;
	switch (device)
	{
	case AUTO:
		type = SeetaDefaultDevice();
		break;
	case CPU:
		break;
	case GPU:
		if (SeetaDefaultDevice() == HOLIDAY_CNN_GPU_DEVICE) type = HOLIDAY_CNN_GPU_DEVICE;
		break;
    default:
        if (device >= self::GPU0 && SeetaDefaultDevice() == HOLIDAY_CNN_GPU_DEVICE)
        {
            type = HOLIDAY_CNN_GPU_DEVICE;
            gpu_id = device - self::GPU0;
        }
	}

	std::string type_string;
	if (type == HOLIDAY_CNN_GPU_DEVICE) type_string = "GPU" + std::to_string(gpu_id);
	else type_string = "CPU";

	orz::Log(orz::INFO) << "Recognizer: " << "Using device " << type_string;

	*recognizer = *other;
	recognizer->model = nullptr;
	recognizer->net = nullptr;

	int err_code;
	if (type == HOLIDAY_CNN_GPU_DEVICE)
	{
		err_code = SeetaCreateNetGPUSharedParam(other->model, GetMaxBatch(), gpu_id, &recognizer->net, &other->param);
	}
	else
	{
		err_code = SeetaCreateNetSharedParam(other->model, GetMaxBatch(), type, &recognizer->net, &other->param);
	}

	if (err_code)
	{
		orz::Log(orz::FATAL) << "Can not init net from unload model" << orz::crash;
	}

	SeetaNetKeepBlob(recognizer->net, recognizer->header.blob_name.c_str());

#ifdef USE_MULTI_THREADS
	recognizer->cores = ExtendMultiNets(recognizer->net, recognizer->GetCoreNumber(), other->model, GetMaxBatch(), type, gpu_id);
	for (auto net : recognizer->cores) SeetaNetKeepBlob(net, recognizer->header.blob_name.c_str());
#endif
}

void VIPLFaceRecognizer::SetNumThreads(int num)
{
	SeetaSetNumThreads(num);
}

VIPLFaceRecognizer::VIPLFaceRecognizer(const char* modelPath)
	: VIPLFaceRecognizer(modelPath, AUTO)
{
}

VIPLFaceRecognizer::VIPLFaceRecognizer(const char* modelPath, Device device)
	: recognizer(new Recognizer)
{
#ifdef SEETA_CHECK_INIT
	SEETA_CHECK_INIT;
#endif
	if (modelPath && !LoadModel(modelPath, device))
	{
		std::cerr << "Error: Can not access \"" << modelPath << "\"!" << std::endl;
		throw std::logic_error("Missing model");
	}
}

VIPLFaceRecognizer::VIPLFaceRecognizer(const char* modelBuffer, size_t bufferLength, Device device)
	: recognizer(new Recognizer)
{
#ifdef SEETA_CHECK_INIT
	SEETA_CHECK_INIT;
#endif
	if (modelBuffer && !LoadModel(modelBuffer, bufferLength, device))
	{
		std::cerr << "Error: Can not initialize from memory!" << std::endl;
		throw std::logic_error("Missing model");
	}
}

VIPLFaceRecognizer::~VIPLFaceRecognizer()
{
	delete recognizer;
}

bool VIPLFaceRecognizer::LoadModel(const char* modelPath)
{
	return LoadModel(modelPath, AUTO);
}

bool VIPLFaceRecognizer::LoadModel(const char* modelPath, Device device)
{
	if (modelPath == NULL) return false;

	// check if an sta file
	{
		auto bin = orz::read_file(modelPath);
		if (bin.empty())
		{
			orz::Log(orz::FATAL) << "Can not access \"" << modelPath << "\"" << orz::crash;
		}

#ifdef SEETA_CHECK_LOAD
		SEETA_CHECK_LOAD(bin);
#endif

		orz::imemorystream infile(bin.data(), bin.size());

		int mark;
		orz::binio<int>::read(infile, mark);
		if (mark == 0x19910929)
		{
			auto model = orz::jug_read(infile);

#ifdef SEETA_CHECK_MODEL
			SEETA_CHECK_MODEL(model, "Using unauthorized model");
#endif

			recognizer->version = std::string(model["version"]);

			if (recognizer->version != "5.0")
			{
				throw orz::Exception("model version missmatch.");
			}

			auto single_data = model["data"];
			if (single_data.valid(orz::Piece::LIST)) single_data = single_data[0];

			auto single_name = model["name"];
			if (single_name.valid(orz::Piece::LIST)) single_name = single_name[0];

			std::string data = std::string(single_data);
			recognizer->date = std::string(model["date"]);
			recognizer->name = std::string(single_name);
			std::string method = std::string(model["trans"]["method"]);

			if (method == "sigmoid")
			{
				float a = model["trans"]["params"][0];
				float b = model["trans"]["params"][1];
				recognizer->trans_func = std::bind(sigmoid, std::placeholders::_1, a, b);
			}
			else if (method == "poly")
			{
				auto params = model["trans"]["params"];
				std::vector<float> ps(params.size());
				for (size_t i = 0; i < params.size(); ++i) ps[i] = params[i];
				recognizer->trans_func = std::bind(poly, std::placeholders::_1, ps);
			}

            auto output = model["output"];
            if (output.valid()){
                recognizer->sqrt_times = orz::jug_get<int>(output["sqrt_times"], -1);
            }

			auto input = model["input"];
			if (input.valid()) {
                recognizer->method = orz::jug_get<std::string>(input["method"], "");
                recognizer->crop_height = orz::jug_get<int>(input["crop_height"], 0);
                recognizer->crop_width = orz::jug_get<int>(input["crop_width"], 0);
			}

			recognizer->fix();

			return LoadModel(data.data(), data.size(), device);
		}
	}
	recognizer->trans_func = nullptr;

	char* buffer = nullptr;
	int64_t buffer_len = 0;
	if (SeetaReadAllContentFromFile(modelPath, &buffer, &buffer_len))
	{
		return false;
	}

	bool loaded = LoadModel(buffer, size_t(buffer_len), device);
	SeetaFreeBuffer(buffer);

	recognizer->fix();

	return loaded;
}

// Header
#ifdef NEED_CHECK
#include "encryption/code/checkit.h"
#endif

bool VIPLFaceRecognizer::LoadModel(const char* modelBuffer, size_t bufferLength, Device device)
{
	// Code
#ifdef NEED_CHECK
	checkit();
#endif

	if (modelBuffer == NULL)
	{
		return false;
	}

	recognizer->free();

    using self = VIPLFaceRecognizer;
	SeetaCNN_DEVICE_TYPE type = HOLIDAY_CNN_CPU_DEVICE;
	int gpu_id = 0;
	switch (device)
	{
	case AUTO:
		type = SeetaDefaultDevice();
		break;
	case CPU:
		break;
	case GPU:
		if (SeetaDefaultDevice() == HOLIDAY_CNN_GPU_DEVICE) type = HOLIDAY_CNN_GPU_DEVICE;
		break;
    default:
        if (device >= self::GPU0 && SeetaDefaultDevice() == HOLIDAY_CNN_GPU_DEVICE)
        {
            type = HOLIDAY_CNN_GPU_DEVICE;
            gpu_id = device - self::GPU0;
        }
	}

	std::string type_string;
	if (type == HOLIDAY_CNN_GPU_DEVICE) type_string = "GPU" + std::to_string(gpu_id);
	else type_string = "CPU";

	orz::Log(orz::INFO) << "Recognizer: " << "Using device " << type_string;

	recognizer->device = device;

	// read header
	size_t header_size = recognizer->header.read(modelBuffer, bufferLength);

	// convert the model
	if (SeetaReadModelFromBuffer(modelBuffer + header_size, bufferLength - header_size, &recognizer->model))
	{
		return false;
	}
	// create the net
	int err_code;
	if (type == HOLIDAY_CNN_GPU_DEVICE)
	{
		err_code = SeetaCreateNetGPUSharedParam(recognizer->model, GetMaxBatch(), gpu_id, &recognizer->net, &recognizer->param);
	}
	else
	{
		err_code = SeetaCreateNetSharedParam(recognizer->model, GetMaxBatch(), type, &recognizer->net, &recognizer->param);
	}

	if (err_code)
	{
		SeetaReleaseModel(recognizer->model);
		recognizer->model = nullptr;
		return false;
	}

	SeetaNetKeepBlob(recognizer->net, recognizer->header.blob_name.c_str());

#ifdef USE_MULTI_THREADS
	recognizer->cores = ExtendMultiNets(recognizer->net, recognizer->GetCoreNumber(), recognizer->model, GetMaxBatch(), type, gpu_id);
	for (auto net : recognizer->cores) SeetaNetKeepBlob(net, recognizer->header.blob_name.c_str());
#endif

	SeetaReleaseModel(recognizer->model);
	recognizer->model = nullptr;

	return true;
}

uint32_t VIPLFaceRecognizer::GetFeatureSize()
{
	return recognizer->header.feature_size;
}

uint32_t VIPLFaceRecognizer::GetCropWidth()
{
	return recognizer->header.width;
}

uint32_t VIPLFaceRecognizer::GetCropHeight()
{
	return recognizer->header.height;
}

uint32_t VIPLFaceRecognizer::GetCropChannels()
{
	return recognizer->header.channels;
}

bool VIPLFaceRecognizer::CropFace(const VIPLImageData& srcImg, const VIPLPoint5& llpoint, const VIPLImageData& dstImg, uint8_t posNum)
{

#ifdef SEETA_CHECK_ANY_FUNCID
	SEETA_CHECK_ANY_FUNCID(std::vector<int>({ 1002, 1003 }), "FaceRecognizer");
#endif
    orz::ctx::lite::bind<orz::Shotgun> _bind_gun(recognizer->m_crop_gun.get());

	float mean_shape[10] = {
		89.3095f, 72.9025f,
		169.3095f, 72.9025f,
		127.8949f, 127.0441f,
		96.8796f, 184.8907f,
		159.1065f, 184.7601f,
	};
	float points[10];
	for (int i = 0; i < 5; ++i)
	{
		points[2 * i] = float(llpoint[i].x);
		points[2 * i + 1] = float(llpoint[i].y);
	}

    if (GetCropHeight() == 256 && GetCropWidth() == 256) {
        face_crop_core(srcImg.data, srcImg.width, srcImg.height, srcImg.channels, dstImg.data, GetCropWidth(), GetCropHeight(), points, 5, mean_shape, 256, 256);
    } else {
        if (recognizer->method == "crop_resize") {
            seeta::Image face256x256(256, 256, 3);
            face_crop_core(srcImg.data, srcImg.width, srcImg.height, srcImg.channels, face256x256.data(), recognizer->crop_width, recognizer->crop_height, points, 5, mean_shape, 256, 256);
            seeta::Image fixed = seeta::resize(face256x256, seeta::Size(GetCropWidth(), GetCropHeight()));
            fixed.copy_to(dstImg.data);
        } else if (recognizer->method == "resize") {
            seeta::Image face256x256(256, 256, 3);
            face_crop_core(srcImg.data, srcImg.width, srcImg.height, srcImg.channels, face256x256.data(), 256, 256, points, 5, mean_shape, 256, 256);
            seeta::Image fixed = seeta::resize(face256x256, seeta::Size(GetCropWidth(), GetCropHeight()));
            fixed.copy_to(dstImg.data);
        } else {
            face_crop_core(srcImg.data, srcImg.width, srcImg.height, srcImg.channels, dstImg.data, GetCropWidth(), GetCropHeight(), points, 5, mean_shape, 256, 256);
        }
    }

	return true;
}

bool VIPLFaceRecognizer::ExtractFeature(const VIPLImageData& cropImg, FaceFeatures const feats)
{
	std::vector<VIPLImageData> faces = { cropImg };
	return ExtractFeature(faces, feats, false);
}

static void normalize(float *features, int num)
{
	double norm = 0;
	float *dim = features;
	for (int i = 0; i < num; ++i)
	{
		norm += *dim * *dim;
		++dim;
	}
	norm = std::sqrt(norm) + 1e-5;
	dim = features;
	for (int i = 0; i < num; ++i)
	{
		*dim /= float(norm);
		++dim;
	}
}

bool VIPLFaceRecognizer::ExtractFeatureNormalized(const VIPLImageData& cropImg, FaceFeatures const feats)
{
	std::vector<VIPLImageData> faces = { cropImg };
	return ExtractFeature(faces, feats, true);
}

bool VIPLFaceRecognizer::ExtractFeatureWithCrop(const VIPLImageData& srcImg, const VIPLPoint5& llpoint, FaceFeatures const feats, uint8_t posNum)
{
	VIPLImageData dstImg(GetCropWidth(), GetCropHeight(), srcImg.channels);
	std::unique_ptr<uint8_t[]> dstImgData(new uint8_t[dstImg.width * dstImg.height * dstImg.channels]);
	dstImg.data = dstImgData.get();
	CropFace(srcImg, llpoint, dstImg, posNum);
	ExtractFeature(dstImg, feats);
	return true;
}

bool VIPLFaceRecognizer::ExtractFeatureWithCropNormalized(const VIPLImageData& srcImg, const VIPLPoint5& llpoint, FaceFeatures const feats, uint8_t posNum)
{
	if (ExtractFeatureWithCrop(srcImg, llpoint, feats, posNum))
	{
		normalize(feats, GetFeatureSize());
		return true;
	}
	return false;
}
#if defined(USE_CBLAS)

#include <cblas.h>

static float cblas_dot(const float* x, const float* y, const long& len) {
    return cblas_sdot(len, x, 1, y, 1);
}

static float cblas_cos(const float* x, const float* y, const long& len) {
    auto dot = cblas_dot(x, y, len);
    auto norm1 = cblas_dot(x, x, len);
    auto norm2 = cblas_dot(y, y, len);
    double similar = dot / (sqrt(norm1 * norm2) + 1e-5);
    return float(similar);
}

#elif defined(USE_SIMD)

#include <xmmintrin.h>
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

static float simd_dot(const float* x, const float* y, const long& len) {
    float inner_prod = 0.0f;
    __m128 X, Y; // 128-bit values
    __m128 acc = _mm_setzero_ps(); // set to (0, 0, 0, 0)
    float temp[4];

    long i;
    for (i = 0; i + 4 <= len; i += 4) {
        X = _mm_loadu_ps(x + i); // load chunk of 4 floats
        Y = _mm_loadu_ps(y + i);
        acc = _mm_add_ps(acc, _mm_mul_ps(X, Y));
    }
    _mm_storeu_ps(&temp[0], acc); // store acc into an array
    inner_prod = temp[0] + temp[1] + temp[2] + temp[3];

    // add the remaining values
    for (; i < len; ++i) {
        inner_prod += x[i] * y[i];
    }
    return inner_prod;
}

static float simd_cos(const float* x, const float* y, const long& len) {
    auto dot = simd_dot(x, y, len);
    auto norm1 = simd_dot(x, x, len);
    auto norm2 = simd_dot(y, y, len);
    double similar = dot / (sqrt(norm1 * norm2) + 1e-5);
    return float(similar);
}

#endif

float VIPLFaceRecognizer::CalcSimilarity(FaceFeatures const fc1, FaceFeatures const fc2, long dim)
{
	if (dim <= 0) dim = GetFeatureSize();
#if defined(USE_CBLAS)
    double similar = cblas_cos(fc1, fc2, dim);
#elif defined(USE_SIMD)
    double similar = simd_cos(fc1, fc2, dim);
#else
	double dot = 0;
	double norm1 = 0;
	double norm2 = 0;
	for (size_t i = 0; i < dim; ++i)
	{
		dot += fc1[i] * fc2[i];
		norm1 += fc1[i] * fc1[i];
		norm2 += fc2[i] * fc2[i];
	}
	double similar = dot / (sqrt(norm1 * norm2) + 1e-5);
#endif

	return recognizer->trans(float(similar));
}

float VIPLFaceRecognizer::CalcSimilarityNormalized(FaceFeatures const fc1, FaceFeatures const fc2, long dim)
{
	if (dim <= 0) dim = GetFeatureSize();
	double dot = 0;
#if defined(USE_CBLAS)
    dot = cblas_dot(fc1, fc2, dim);
#elif defined(USE_SIMD)
    dot = simd_dot(fc1, fc2, dim);
#else

	float *fc1_dim = fc1;
	float *fc2_dim = fc2;
	for (int i = 0; i < dim; ++i)
	{
		dot += *fc1_dim * *fc2_dim;
		++fc1_dim;
		++fc2_dim;
	}
#endif

	double similar = dot;
	return recognizer->trans(float(similar));
}

int VIPLFaceRecognizer::SetMaxBatchGlobal(int max_batch)
{
	std::swap(max_batch, Recognizer::max_batch_global);
	return max_batch;
}

int VIPLFaceRecognizer::GetMaxBatch()
{
	return recognizer->GetMaxBatch();
}

int VIPLFaceRecognizer::SetCoreNumberGlobal(int core_number)
{
	std::swap(core_number, Recognizer::core_number_global);
	return core_number;

}

int VIPLFaceRecognizer::GetCoreNumber()
{
	return recognizer->GetCoreNumber();
}

template <typename T>
static void CopyData(T *dst, const T *src, size_t count)
{
#if _MSC_VER >= 1600
	memcpy_s(dst, count * sizeof(T), src, count * sizeof(T));
#else
	memcpy(dst, src, count * sizeof(T));
#endif
}

static bool LocalExtractFeature(
	int number, int width, int height, int channels, unsigned char *data,
	SeetaCNN_Net *net, int max_batch, const char *blob_name, int feature_size,
	float *feats,
	bool normalization, int sqrt_times = 0)
{
#ifdef USE_MULTI_THREADS
#else
#ifdef SEETA_CHECK_ANY_FUNCID
	SEETA_CHECK_ANY_FUNCID(std::vector<int>({ 1002, 1003 }), "FaceRecognizer");
#endif
#endif
	if (!net) return false;
	if (data == nullptr || number <= 0) return true;

	auto single_image_size = channels * height * width;

	if (number > max_batch)
	{
		// Divide and Conquer
		int end = number;
		int step = max_batch;
		int left = 0;
		while (left < end)
		{
			int right = std::min(left + step, end);

			int local_number = right - left;
			unsigned char *local_data = data + left * single_image_size;
			float *local_feats = feats + left * feature_size;

			if (!LocalExtractFeature(
				local_number, width, height, channels, local_data,
				net, max_batch, blob_name, feature_size,
				local_feats,
				normalization,
                sqrt_times
				)) return false;
			left = right;
		}
		return true;
	}

	SeetaCNN_InputOutputData himg;
	himg.number = number;
	himg.channel = channels;
	himg.height = height;
	himg.width = width;
	himg.buffer_type = SEETACNN_BGR_IMGE_CHAR;
	himg.data_point_char = data;

	// do forward
	if (SeetaRunNetChar(net, 1, &himg))
	{
		ORZ_LOG(orz::INFO) << "SeetaRunNetChar failed.";
		return false;
	}

	// get the output
	SeetaCNN_InputOutputData output;
	if (SeetaGetFeatureMap(net, blob_name, &output))
	{

		ORZ_LOG(orz::INFO) << "SeetaGetFeatureMap failed.";
		return false;
	}

	// check the output size
	if (output.channel * output.height * output.width != feature_size || output.number != himg.number)
	{

		ORZ_LOG(orz::INFO) << "output shape missmatch. " << feature_size << " expected. but " << output.channel * output.height * output.width << " given";
		return false;
	}

	// copy data for output
	CopyData(feats, output.data_point_float, output.number * feature_size);

	int32_t all_feats_size = output.number * feature_size;
	float *all_feats = feats;

#if defined(DOUBLE_SQRT) || defined(SINGLE_SQRT)
	for (int i = 0; i != all_feats_size; i++)
	{
#if defined(DOUBLE_SQRT)
		feat[i] = sqrt(sqrt(feat[i]));
#elif defined(SINGLE_SQRT)
		all_feats[i] = sqrt(all_feats[i]);
#endif // DOUBLE_SQRT
	}
#endif // DOUBLE_SQRT || SINGLE_SQRT

    if (sqrt_times > 0) {
        while (sqrt_times--) {
            for (int i = 0; i != all_feats_size; i++) all_feats[i] = std::sqrt(all_feats[i]);
        }
    }

	if (normalization)
	{
		for (int i = 0; i < number; ++i)
		{
			float *local_feats = feats + i * feature_size;
			normalize(local_feats, feature_size);
		}
	}

	return true;
}

static bool LocalExtractFeatureThreads(
	int number, int width, int height, int channels, unsigned char *data,
	std::vector<SeetaCNN_Net *> &cores, orz::Shotgun &gun,
	int max_batch, const char *blob_name, int feature_size,
	float *feats,
	bool normalization, int sqrt_times = 0)
{
#ifdef SEETA_CHECK_ANY_FUNCID
	SEETA_CHECK_ANY_FUNCID(std::vector<int>({ 1002, 1003 }), "FaceRecognizer");
#endif

	if (data == nullptr || number <= 0) return true;

	auto single_image_size = channels * height * width;

	// Divide and Conquer
	int end = number;
	int step = max_batch;
	int left = 0;
	while (left < end)
	{
		int right = std::min(left + step, end);

		int local_number = right - left;
		unsigned char *local_data = data + left * single_image_size;
		float *local_feats = feats + left * feature_size;

		gun.fire([=, &cores](int id)
		{
			auto net = cores[id];
			LocalExtractFeature(
				local_number, width, height, channels, local_data,
				net, max_batch, blob_name, feature_size,
				local_feats,
				normalization,
				sqrt_times
				);
		});

		left = right;
	}
	gun.join();
	return true;
}

bool VIPLFaceRecognizer::ExtractFeature(const std::vector<VIPLImageData>& faces, float* feats, bool normalization)
{
    orz::ctx::lite::bind<orz::Shotgun> _bind_gun(recognizer->m_crop_gun.get());

	if (!recognizer->net) return false;
	if (faces.empty()) return true;

	int number = int(faces.size());
	int channels = GetCropChannels();
	int height = GetCropHeight();
	int width = GetCropWidth();

	// check all size
//	for (int i = 0; i < faces.size(); ++i)
//	{
//		if (faces[i].channels != channels ||
//			faces[i].height != height ||
//			faces[i].width != width
//			) return false;
//	}

	auto single_image_size = channels * height * width;
	std::unique_ptr<unsigned char[]> data_point_char(new unsigned char[number * single_image_size]);
	for (int i = 0; i < number; ++i)
	{

		if (faces[i].channels == channels && 
			faces[i].height == height &&
			faces[i].width == width)
		{
			CopyData(&data_point_char[i * single_image_size], faces[i].data, single_image_size);
			continue;
        }

        if (recognizer->method == "crop_resize") {
            seeta::Image face(faces[i].data, faces[i].width, faces[i].height, faces[i].channels);
            seeta::Rect rect((recognizer->crop_width - faces[i].width) / 2, (recognizer->crop_height - faces[i].height) / 2, recognizer->crop_width, recognizer->crop_height);
            seeta::Image fixed = seeta::crop_resize(face, rect, seeta::Size(GetCropWidth(), GetCropHeight()));
            CopyData(&data_point_char[i * single_image_size], fixed.data(), single_image_size);
        } else if (recognizer->method == "resize") {
            seeta::Image face(faces[i].data, faces[i].width, faces[i].height, faces[i].channels);
            seeta::Image fixed = seeta::resize(face, seeta::Size(GetCropWidth(), GetCropHeight()));
            CopyData(&data_point_char[i * single_image_size], fixed.data(), single_image_size);
        }
		else {
			seeta::Image face(faces[i].data, faces[i].width, faces[i].height, faces[i].channels);
			seeta::Rect rect((GetCropWidth() - faces[i].width) / 2, (GetCropHeight() - faces[i].height) / 2, GetCropWidth(), GetCropHeight());
			seeta::Image fixed = seeta::crop_resize(face, rect, seeta::Size(GetCropWidth(), GetCropHeight()));
			CopyData(&data_point_char[i * single_image_size], fixed.data(), single_image_size);
		}

	}
#ifdef USE_MULTI_THREADS
	if (number <= GetMaxBatch() || GetCoreNumber() <= 1)
	{
		return LocalExtractFeature(
			number, width, height, channels, data_point_char.get(),
			recognizer->net, GetMaxBatch(), recognizer->header.blob_name.c_str(), GetFeatureSize(),
			feats,
			normalization,
			recognizer->sqrt_times);
	}
	return LocalExtractFeatureThreads(
		number, width, height, channels, data_point_char.get(),
		recognizer->cores, *recognizer->gun, GetMaxBatch(), recognizer->header.blob_name.c_str(), GetFeatureSize(),
		feats,
		normalization,
		recognizer->sqrt_times);
#else
	return LocalExtractFeature(
		number, width, height, channels, data_point_char.get(),
		recognizer->net, GetMaxBatch(), recognizer->header.blob_name.c_str(), GetFeatureSize(),
		feats,
		normalization,
		recognizer->sqrt_times);
#endif
}

bool VIPLFaceRecognizer::ExtractFeatureNormalized(const std::vector<VIPLImageData>& faces, float* feats)
{
	return ExtractFeature(faces, feats, true);
}

// on checking param, sure right
static bool CropFaceThreads(VIPLFaceRecognizer &FR, const std::vector<VIPLImageData>& images,
	const std::vector<VIPLPoint>& points, unsigned char *faces_data, orz::Shotgun &gun)
{
	const int PN = 5;
	const auto single_image_size = FR.GetCropChannels() * FR.GetCropHeight() * FR.GetCropWidth();
	// unsigned char *single_face_data = faces_data;
	// const VIPLPoint *single_points = points.data();

	for (size_t i = 0; i < images.size(); ++i)
	{
		gun.fire([&, i](int id)
		{
			const auto single_face_data = faces_data + i * single_image_size;
			const auto single_points = points.data() + i * PN;

			VIPLImageData face(FR.GetCropWidth(), FR.GetCropHeight(), FR.GetCropChannels());
			face.data = single_face_data;

			VIPLPoint5 local_points;
			CopyData(local_points, single_points, PN);

			FR.CropFace(images[i], local_points, face);
		});

		// single_points += PN;
		// single_face_data += single_image_size;
	}
	gun.join();
	return true;
}


// on checking param, sure right
static bool CropFaceBatch(VIPLFaceRecognizer &FR, const std::vector<VIPLImageData>& images,
	const std::vector<VIPLPoint>& points, unsigned char *faces_data)
{
	const int PN = 5;
	const auto single_image_size = FR.GetCropChannels() * FR.GetCropHeight() * FR.GetCropWidth();
	unsigned char *single_face_data = faces_data;
	const VIPLPoint *single_points = points.data();
	for (size_t i = 0; i < images.size(); ++i)
	{
		VIPLImageData face(FR.GetCropWidth(), FR.GetCropHeight(), FR.GetCropChannels());
		face.data = single_face_data;

		VIPLPoint5 local_points;
		CopyData(local_points, single_points, PN);

		if (!FR.CropFace(images[i], local_points, face)) return false;

		single_points += PN;
		single_face_data += single_image_size;
	}
	return true;
}

bool VIPLFaceRecognizer::ExtractFeatureWithCrop(const std::vector<VIPLImageData>& images,
	const std::vector<VIPLPoint>& points, float* feats, bool normalization)
{
    orz::ctx::lite::bind<orz::Shotgun> _bind_gun(recognizer->m_crop_gun.get());

	if (!recognizer->net) return false;
	if (images.empty()) return true;

	const int PN = 5;

    if (images.size() * PN != points.size())
	{
		return false;
	}

	// crop face
	std::unique_ptr<unsigned char[]> faces_data(new unsigned char[images.size() * GetCropChannels() * GetCropHeight() * GetCropWidth()]);
#ifdef USE_MULTI_THREADS
	if (images.size() <= 1 || GetCoreNumber() <= 1) ::CropFaceBatch(*this, images, points, faces_data.get());
	else
	::CropFaceThreads(*this, images, points, faces_data.get(), *recognizer->gun);
#else
	::CropFaceBatch(*this, images, points, faces_data.get());
#endif

	int number = int(images.size());
	int channels = GetCropChannels();
	int height = GetCropHeight();
	int width = GetCropWidth();
#ifdef USE_MULTI_THREADS
	if (number <= GetMaxBatch() || GetCoreNumber() <= 1)
	{
		return LocalExtractFeature(
			number, width, height, channels, faces_data.get(),
			recognizer->net, GetMaxBatch(), recognizer->header.blob_name.c_str(), GetFeatureSize(),
			feats,
			normalization,
			recognizer->sqrt_times);
	}
	return LocalExtractFeatureThreads(
		number, width, height, channels, faces_data.get(),
		recognizer->cores, *recognizer->gun, GetMaxBatch(), recognizer->header.blob_name.c_str(), GetFeatureSize(),
		feats,
		normalization,
		recognizer->sqrt_times
		);
#else
	return LocalExtractFeature(
		number, width, height, channels, faces_data.get(),
		recognizer->net, GetMaxBatch(), recognizer->header.blob_name.c_str(), GetFeatureSize(),
		feats,
		normalization,
		recognizer->sqrt_times
		);
#endif
}

bool VIPLFaceRecognizer::ExtractFeatureWithCropNormalized(const std::vector<VIPLImageData>& images,
	const std::vector<VIPLPoint>& points, float* feats)
{
	return ExtractFeatureWithCrop(images, points, feats, true);
}
