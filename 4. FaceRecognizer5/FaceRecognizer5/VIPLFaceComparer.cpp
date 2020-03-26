#include "VIPLFaceComparer.h"

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

class VIPLFaceComparer::Data
{
public:
	seeta::FRModelHeader header;

	std::string version;
	std::string date;
	std::string name;
	std::function<float(float)> trans_func;

	// for YuLe setting
    int sqrt_times = -1;
	std::string default_method = "crop";
    std::string method = "";
    int crop_width = 0;
    int crop_height = 0;
    
    Data()
    {
        header.width = 256;
        header.height = 256;
        header.channels = 3;
    }

	void free()
	{
	}

	float trans(float similar) const
	{
		if (trans_func)
		{
			return trans_func(similar);
		}
		return similar;
	}

    ~Data()
	{
        this->free();
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

static float sigmoid(float x, float a = 0, float b = 1)
{
    return 1 / (1 + exp(a - b * x));
}

static float poly(float x, const std::vector<float> &params)
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

VIPLFaceComparer::VIPLFaceComparer(const char* modelPath)
	: data(new Data)
{
#ifdef SEETA_CHECK_INIT
	SEETA_CHECK_INIT;
#endif
	if (modelPath && !LoadModel(modelPath))
	{
		std::cerr << "Error: Can not access \"" << modelPath << "\"!" << std::endl;
        delete data;
		throw std::logic_error("Missing model");
	}
}

VIPLFaceComparer::~VIPLFaceComparer()
{
    delete data;
}


bool VIPLFaceComparer::LoadModel(const char* modelPath)
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

            this->data->version = std::string(model["version"]);

            if (this->data->version != "5.0")
			{
				throw orz::Exception("model version missmatch.");
			}

			auto single_data = model["data"];
			if (single_data.valid(orz::Piece::LIST)) single_data = single_data[0];

			auto single_name = model["name"];
			if (single_name.valid(orz::Piece::LIST)) single_name = single_name[0];

			std::string data = std::string(single_data);
			this->data->date = std::string(model["date"]);
            this->data->name = std::string(single_name);
			std::string method = std::string(model["trans"]["method"]);

			if (method == "sigmoid")
			{
				float a = model["trans"]["params"][0];
				float b = model["trans"]["params"][1];
                this->data->trans_func = std::bind(sigmoid, std::placeholders::_1, a, b);
			}
			else if (method == "poly")
			{
				auto params = model["trans"]["params"];
				std::vector<float> ps(params.size());
				for (size_t i = 0; i < params.size(); ++i) ps[i] = params[i];
                this->data->trans_func = std::bind(poly, std::placeholders::_1, ps);
			}

            auto output = model["output"];
            if (output.valid()){
                this->data->sqrt_times = orz::jug_get<int>(output["sqrt_times"], -1);
            }

			auto input = model["input"];
			if (input.valid()) {
                this->data->method = orz::jug_get<std::string>(input["method"], "");
                this->data->crop_height = orz::jug_get<int>(input["crop_height"], 0);
                this->data->crop_width = orz::jug_get<int>(input["crop_width"], 0);
			}

            this->data->fix();

            this->data->header.read(data.data(), data.size());

            return true;
		}
	}
    this->data->trans_func = nullptr;

    char* buffer = nullptr;
    int64_t buffer_len = 0;
    if (SeetaReadAllContentFromFile(modelPath, &buffer, &buffer_len))
    {
        return false;
    }
    this->data->header.read(buffer, size_t(buffer_len));
    SeetaFreeBuffer(buffer);

    this->data->fix();


	return true;
}

// Header
#ifdef NEED_CHECK
#include "encryption/code/checkit.h"
#endif

uint32_t VIPLFaceComparer::GetCropWidth()
{
	return data->header.width;
}

uint32_t VIPLFaceComparer::GetCropHeight()
{
    return data->header.height;
}

uint32_t VIPLFaceComparer::GetCropChannels()
{
    return data->header.channels;
}

bool VIPLFaceComparer::CropFace(const VIPLImageData& srcImg, const VIPLPoint (&llpoint)[5], const VIPLImageData& dstImg, uint8_t posNum)
{

#ifdef SEETA_CHECK_ANY_FUNCID
	SEETA_CHECK_ANY_FUNCID(std::vector<int>({ 1002, 1003 }), "FaceRecognizer");
#endif

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
        if (data->method == "crop_resize") {
            seeta::Image face256x256(256, 256, 3);
            face_crop_core(srcImg.data, srcImg.width, srcImg.height, srcImg.channels, face256x256.data(), data->crop_width, data->crop_height, points, 5, mean_shape, 256, 256);
            seeta::Image fixed = seeta::resize(face256x256, seeta::Size(GetCropWidth(), GetCropHeight()));
            fixed.copy_to(dstImg.data);
        } else if (data->method == "resize") {
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

float VIPLFaceComparer::CalcSimilarity(const float *fc1, const float *fc2, long dim)
{
	if (dim <= 0) dim = data->header.feature_size;
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

    return data->trans(float(similar));
}

float VIPLFaceComparer::CalcSimilarityNormalized(const float *fc1, const float *fc2, long dim)
{
    if (dim <= 0) dim = data->header.feature_size;
	double dot = 0;
#if defined(USE_CBLAS)
    dot = cblas_dot(fc1, fc2, dim);
#elif defined(USE_SIMD)
    dot = simd_dot(fc1, fc2, dim);
#else

	const float *fc1_dim = fc1;
	const float *fc2_dim = fc2;
	for (int i = 0; i < dim; ++i)
	{
		dot += *fc1_dim * *fc2_dim;
		++fc1_dim;
		++fc2_dim;
	}
#endif

	double similar = dot;
    return data->trans(float(similar));
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
