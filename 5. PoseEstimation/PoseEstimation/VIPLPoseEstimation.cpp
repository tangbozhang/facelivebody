#include "VIPLPoseEstimation.h"

#define VIPL_SUPPORT
#include "seeta/ImageProcess.h"
#include "seeta/ForwardNet.h"

#include <cmath>
#include <iostream>
#include "seeta/ModelHeader.h"

#include <orz/io/i.h>
#include <orz/utils/log.h>
//#include "orz_resources.h"

#define VER_HEAD(x) #x "."
#define VER_TAIL(x) #x
#define GENERATE_VER(seq) FUN_MAJOR seq
#define FUN_MAJOR(x) VER_HEAD(x) FUN_MINOR
#define FUN_MINOR(x) VER_HEAD(x) FUN_SINOR
#define FUN_SINOR(x) VER_TAIL(x)

#define LIBRARY_VERSION GENERATE_VER( \
	(VIPL_POSE_ESTIMATION_MAJOR_VERSION) \
	(VIPL_POSE_ESTIMATION_MINOR_VERSION) \
	(VIPL_POSE_ESTIMATION_SUBMINOR_VERSION))

#define LIBRARY_NAME "PoseEstimator"

#define LOG_HEAD LIBRARY_NAME "(" LIBRARY_VERSION "): "

class VIPLPoseEstimationCore
{
public:
    ForwardNet net;

    bool LoadNet(const char *data, size_t size, VIPLPoseEstimation::Device device)
    {

        if (data == nullptr || size == 0)
        {
#ifdef USE_ORZ_RESOURCES
            static const char *model_url = "@/model/VIPLPoseEstimation1.1.0.ext.dat";
            orz::Log(orz::STATUS) << LOG_HEAD << "Loading resources: " << model_url;

            auto model = orz_resources_get(model_url);

            if (model.data == nullptr || model.size == 0)
            {
                orz::Log(orz::FATAL) << LOG_HEAD << "Can not access resources: " << model_url << orz::crash;
            }
            data = model.data;
            size = model.size;
#else
            orz::Log(orz::FATAL) << "Can not initialize object without specific model path." << orz::crash;
#endif
        }

        using self = VIPLPoseEstimation;
        SeetaCNN_DEVICE_TYPE type = HOLIDAY_CNN_CPU_DEVICE;
        int gpu_id = 0;
        switch (device)
        {
        case self::AUTO:
            type = SeetaDefaultDevice();
            break;
        case self::CPU:
            break;
        case self::GPU:
            if (SeetaDefaultDevice() == HOLIDAY_CNN_GPU_DEVICE) type = HOLIDAY_CNN_GPU_DEVICE;
            break;
        default:
            if (device >= self::GPU0
                && SeetaDefaultDevice() == HOLIDAY_CNN_GPU_DEVICE)
            {
                type = HOLIDAY_CNN_GPU_DEVICE;
                gpu_id = device - self::GPU0;
            }
        }

        std::string type_string;
        if (type == HOLIDAY_CNN_GPU_DEVICE) type_string = "GPU" + std::to_string(gpu_id);
        else type_string = "CPU";

        orz::Log(orz::INFO) << LOG_HEAD << "Using device " << type_string;

        seeta::ModelHeader header;
        int header_size = header.read(data, size);

        bool succeed = this->net.LoadModel(data + header_size, size - header_size, 1,
            header.channels, header.height, header.width,
            header.blob_name,
            ForwardNet::Device(type), gpu_id);

        return succeed;
    }
};

VIPLPoseEstimation::VIPLPoseEstimation(const char* model_path)
	: VIPLPoseEstimation(model_path, AUTO)
{
}

// Header
#ifdef NEED_CHECK
#include "encryption/code/checkit.h"
#endif

#if SEETA_LOCK_SDK
#include <lock/macro.h>
#endif

VIPLPoseEstimation::VIPLPoseEstimation(const char* model_path, Device device)
	: impl(new VIPLPoseEstimationCore)
{
// Code
#ifdef NEED_CHECK
	checkit(model_path);
#endif

#ifdef SEETA_CHECK_INIT
	SEETA_CHECK_INIT;
#endif

    if (model_path == nullptr)
    {
        if (!impl->LoadNet(nullptr, 0, device))
        {
            orz::Log(orz::FATAL) << "Load model in resources failed." << orz::crash;
        }
        return;
    }

	auto bin = orz::read_file(model_path);
	if (bin.size() == 0)
	{
		orz::Log(orz::FATAL) << "Can not access \"" << model_path << "\"" << orz::crash;
	}

#ifdef SEETA_CHECK_LOAD
	SEETA_CHECK_LOAD(bin);
#endif

    if (!impl->LoadNet(bin.data<char>(), bin.size(), device))
	{
		orz::Log(orz::FATAL) << "Got a broken model \"" << model_path << "\"" << orz::crash;
    }
}

VIPLPoseEstimation::VIPLPoseEstimation(Device device)
    : impl(new VIPLPoseEstimationCore)
{
    // Code
#ifdef NEED_CHECK
    checkit();
#endif

#ifdef SEETA_CHECK_INIT
    SEETA_CHECK_INIT;
#endif

    if (!impl->LoadNet(nullptr, 0, device))
    {
        orz::Log(orz::FATAL) << "Load model in resources failed." << orz::crash;
    }
}

VIPLPoseEstimation::~VIPLPoseEstimation()
{
}

static float raw2degree(float raw)
{
	return float((1.0 / (1.0 + std::exp(-raw)))*180.0 - 90.0);
}

bool VIPLPoseEstimation::Estimate(const VIPLImageData& src_img, const VIPLFaceInfo& info, float& yaw, float& pitch, float& roll)
{
#ifdef SEETA_CHECK_AUTO_FUNCID
	SEETA_CHECK_AUTO_FUNCID("PoseEstimation");
#endif

	yaw = pitch = roll = 0;
	if (!impl->net.valid() || !src_img.data || src_img.channels != 3)
	{
		return false;
	}
	// 首先取出人脸并crop成90x90
	seeta::Rect rect(info.x, info.y, info.width, info.height);
	seeta::Size size(impl->net.GetInputWidth(), impl->net.GetInputHeight());

	seeta::Image cropped_face = seeta::crop_resize(src_img, rect, size);

	// 执行前向
    auto output = impl->net.Forward(cropped_face);
    
    if (output.count() != 3)
	{
		return false;
	}

#ifdef _DEBUG
	int _count = 3;
	std::cout << "LOG: Predict count: " << _count << std::endl;
	std::cout << "LOG: Predict result: ";
	for (int i = 0; i < _count; ++i)
	{
		if (i) std::cout << ", ";
		std::cout << output[i];
	}
	std::cout << std::endl;
#endif	// _DEBUG

	yaw = raw2degree(output[0]);
	pitch = raw2degree(output[1]);
	roll = raw2degree(output[2]);

	return true;
}
