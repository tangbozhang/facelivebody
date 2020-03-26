#include "VIPLEyeBlinkDetector.h"

#define VIPL_SUPPORT
#include "seeta/ForwardNet.h"
#include "seeta/ImageProcess.h"
#include <iostream>
#include <orz/utils/log.h>
#include <orz/io/i.h>
#include <cmath>

class VIPLEyeBlinkDetectorCore
{
public:
	ForwardNet net;


	// return max index, 0 for close, 1 for open, 2 for not
	int Detect(const seeta::Image &image)
	{
		auto output = this->net.Forward(image);
		int count = output.count();
		int max = 0;
		for (int i = 1; i < count; ++i)
		{
			// std::cout << output.data(i) << " ";
			if (output.data(i) > output.data(max)) max = i;
		}
		// std::cout << std::endl;
		return max;
	}

	// return max index, 0 for close, 1 for open, 2 for not
	std::vector<int> Detect(const std::vector<seeta::Image> &images)
	{
		auto output = this->net.Forward(images);
		int number = output.shape(0);
		int count = output.shape(1) * output.shape(2) * output.shape(3);
		
		std::vector<int> maxs(number, 0);
		for (int i = 0; i < number; ++i)
		{
			int max = 0;
			for (int c = 1; c < count; ++c)
			{
				if (output.data(i, 0, 0, c) > output.data(i, 0, 0, max)) max = c;
			}
			maxs[i] = max;
		}
		return std::move(maxs);
	}
};

#ifdef NEED_CHECK
#include "encryption/code/checkit.h"
#endif

#if SEETA_LOCK_SDK
#include <lock/macro.h>
#endif

VIPLEyeBlinkDetector::VIPLEyeBlinkDetector(const char* model_path, Device device)
	: impl(new VIPLEyeBlinkDetectorCore)
{
	// Code
#ifdef NEED_CHECK
	checkit();
#endif

#ifdef SEETA_CHECK_INIT
	SEETA_CHECK_INIT;
#endif

    using self = VIPLEyeBlinkDetector;
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
        if (device >= self::GPU0 && device <= self::GPU7
            && SeetaDefaultDevice() == HOLIDAY_CNN_GPU_DEVICE)
        {
            type = HOLIDAY_CNN_GPU_DEVICE;
            gpu_id = device - self::GPU0;
        }
    }

    std::string type_string;
    if (type == HOLIDAY_CNN_GPU_DEVICE) type_string = "GPU" + std::to_string(gpu_id);
    else type_string = "CPU";

    orz::Log(orz::INFO) << "EyeBlinkDetector: " << "Using device " << type_string;

	if (!impl->net.LoadModel(model_path, 2, 1, 99, 99, "prob", ForwardNet::Device(type), gpu_id))
	{
		std::cerr << "Error: " << "Can not access \"" << model_path << "\"" << std::endl;
		throw std::logic_error("Missing model");
	}
}

VIPLEyeBlinkDetector::~VIPLEyeBlinkDetector()
{
}

int VIPLEyeBlinkDetector::Detect(const VIPLImageData& image, const VIPLPoint* points)
{
#ifdef SEETA_CHECK_AUTO_FUNCID
	SEETA_CHECK_AUTO_FUNCID("EyeBlinkDetector");
#endif

	if (!points) return 0;

	seeta::Image simage = image;

	auto shape = seeta::face_meanshape(5, 0);
	seeta::Landmarks cropMarks;

	auto cropImage = seeta::crop_face(simage, shape, seeta::Landmarks(points, 5), seeta::BY_BICUBIC, shape.size, cropMarks);

	double eyeSpan = sqrt(pow((cropMarks.points[1].x - cropMarks.points[0].x), 2) + pow((cropMarks.points[1].y - cropMarks.points[0].y), 2));
	seeta::Size eyeSize(99, 99);

	int cropSize = cropImage.height();
	//防止边界溢出
	int leftEyePointX = std::max(int(cropMarks.points[0].x - eyeSpan / 2), 0);
	int leftEyePointY = std::max(int(cropMarks.points[0].y - eyeSpan / 2), 0);
	leftEyePointX = std::min(int(cropSize - eyeSpan / 2), leftEyePointX);
	leftEyePointY = std::min(int(cropSize - eyeSpan / 2), leftEyePointY);

	int rightEyePointX = std::max(int(cropMarks.points[1].x - eyeSpan / 2), 0);
	int rightEyePointY = std::max(int(cropMarks.points[1].y - eyeSpan / 2), 0);
	rightEyePointX = std::min(int(cropSize - eyeSpan / 2), rightEyePointX);
	rightEyePointY = std::min(int(cropSize - eyeSpan / 2), rightEyePointY);

	seeta::Image leftEye = seeta::crop_resize(cropImage, seeta::Rect(leftEyePointX, leftEyePointY, eyeSpan / 1, eyeSpan / 1), eyeSize);
	seeta::Image rightEye = seeta::crop_resize(cropImage, seeta::Rect(rightEyePointX, rightEyePointY, eyeSpan / 1, eyeSpan / 1), eyeSize);

	seeta::Image gray_leftEye = seeta::gray(leftEye);
	seeta::Image gray_rightEye = seeta::gray(rightEye);


	gray_leftEye = seeta::equalize_hist(gray_leftEye);
	gray_rightEye = seeta::equalize_hist(gray_rightEye);

	auto eyeStatus = impl->Detect({ gray_leftEye, gray_rightEye });

	auto leftEyeStatus = eyeStatus[0];
	auto rightEyeStatus = eyeStatus[1];

	int old_status = (leftEyeStatus == 0 ? 1 : 0) | ((rightEyeStatus == 0 ? 1 : 0) << 1);

	return old_status;

}

bool VIPLEyeBlinkDetector::ClosedEyes(const VIPLImageData& image, const VIPLPoint* points)
{
#ifdef SEETA_CHECK_AUTO_FUNCID
	SEETA_CHECK_AUTO_FUNCID("EyeBlinkDetector");
#endif

	// 注意：此版本闭眼的检测率比睁眼高，
	if (!points) return 0;

	seeta::Image simage = image;

	auto shape = seeta::face_meanshape(5, 0);
	seeta::Landmarks cropMarks;

	auto cropImage = seeta::crop_face(simage, shape, seeta::Landmarks(points, 5), seeta::BY_BICUBIC, shape.size, cropMarks);

	double eyeSpan = sqrt(pow((cropMarks.points[1].x - cropMarks.points[0].x), 2) + pow((cropMarks.points[1].y - cropMarks.points[0].y), 2));
	seeta::Size eyeSize(99, 99);
	int cropSize = cropImage.height();
	//防止边界溢出
	int leftEyePointX = std::max(int(cropMarks.points[0].x - eyeSpan / 2), 0);
	int leftEyePointY = std::max(int(cropMarks.points[0].y - eyeSpan / 2), 0);
	leftEyePointX = std::min(int(cropSize - eyeSpan / 2), leftEyePointX);
	leftEyePointY = std::min(int(cropSize - eyeSpan / 2), leftEyePointY);

	int rightEyePointX = std::max(int(cropMarks.points[1].x - eyeSpan / 2), 0);
	int rightEyePointY = std::max(int(cropMarks.points[1].y - eyeSpan / 2), 0);
	rightEyePointX = std::min(int(cropSize - eyeSpan / 2), rightEyePointX);
	rightEyePointY = std::min(int(cropSize - eyeSpan / 2), rightEyePointY);

	seeta::Image leftEye = seeta::crop_resize(cropImage, seeta::Rect(leftEyePointX, leftEyePointY, eyeSpan / 1, eyeSpan / 1), eyeSize);
	seeta::Image rightEye = seeta::crop_resize(cropImage, seeta::Rect(rightEyePointX, rightEyePointY, eyeSpan / 1, eyeSpan / 1), eyeSize);

	seeta::Image gray_leftEye = seeta::gray(leftEye);
	seeta::Image gray_rightEye = seeta::gray(rightEye);

	gray_leftEye = seeta::equalize_hist(gray_leftEye);
	gray_rightEye = seeta::equalize_hist(gray_rightEye);

	auto eyeStatus = impl->Detect({ gray_leftEye, gray_rightEye });

	if (eyeStatus[0] == 2 && eyeStatus[1] == 2) return false;

	// bool closed = true;
	// if (eyeStatus[0] != 2) closed = closed && eyeStatus[0] == 0;
	// if (eyeStatus[1] != 2) closed = closed && eyeStatus[1] == 0;

	bool closed = eyeStatus[0] == 0 && eyeStatus[1] == 0;
	
	return closed;
}
