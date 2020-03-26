#include "VIPLQualityAssessment.h"
#include <VIPLPoseEstimation.h>
#include "seeta/ImageProcess.h"

#include <cfloat>
#include <memory>
#include <complex>

//#include <orz/utils/log.h>

float Evaluate(const VIPLPoint* points, int num)
{
	return 0;
}

class VIPLQualityAssessmentCore
{
public:
	VIPLQualityAssessmentCore(int satisfied_face_size, int min_face_size)
		: satisfied_face_size(satisfied_face_size), min_face_size(min_face_size)
	{}
	int satisfied_face_size;
	int min_face_size;
	std::shared_ptr<VIPLPoseEstimation>  pose;
	float clarity_threshold = 0.4;
};

// Header
#ifdef NEED_CHECK
#include "encryption/code/checkit.h"
#endif

std::string ptr(const void *p)
{
	std::ostringstream oss;
	oss << std::showbase;
	oss << "0x" << std::hex << p;
	oss << std::noshowbase;
	return oss.str();
}

VIPLQualityAssessment::VIPLQualityAssessment(const char* model_path, int satisfied_face_size)
	: impl(new VIPLQualityAssessmentCore(satisfied_face_size, 64))
{
	// orz::Log(orz::INFO) << "QualityAssessment(" << ptr(this) << "): " << "init.";
	// Code
#ifdef NEED_CHECK
	checkit();
#endif // RELEASE_DOG || TIME_LOCK

	// orz::Log(orz::INFO) << "QualityAssessment(" << ptr(this) << "): " << "loading model \"" << model_path << "\"";
	// orz::Log(orz::INFO) << "QualityAssessment(" << ptr(this) << "): " << "satisfied face size: " << satisfied_face_size;

	setSatisfiedFaceSize(satisfied_face_size);
	loadPoseEstimationModel(model_path);
}

void VIPLQualityAssessment::setSatisfiedFaceSize(int size)
{
	const int MIN_FACE_WIDTH = 64;
	impl->satisfied_face_size = std::max(MIN_FACE_WIDTH + 1, size);
}

int VIPLQualityAssessment::getSatisfiedFaceSize() const
{
	return impl->satisfied_face_size;
}

void VIPLQualityAssessment::setClarityThreshold(int thresh)
{
	impl->clarity_threshold = thresh;
}

int VIPLQualityAssessment::getClarityThreshold() const
{
	return impl->clarity_threshold;
}

bool VIPLQualityAssessment::loadPoseEstimationModel(const char* model_path)
{
	try
	{
		impl->pose.reset(new VIPLPoseEstimation(model_path));

		// orz::Log(orz::INFO) << "QualityAssessment(" << ptr(this) << "): " << "load model succeed.";
		return true;
	}
	catch (...)
	{

		// orz::Log(orz::INFO) << "QualityAssessment(" << ptr(this) << "): " << "load model failed.";
		return false;
	}
}


float VIPLQualityAssessment::Evaluate(const VIPLImageData& srcImg, const VIPLFaceInfo& faceInfo)
{
	//orz::Log(orz::INFO) << "QualityAssessment(" << ptr(this) << "): " << "Evaluate.";
	//orz::Log(orz::INFO) << "QualityAssessment(" << ptr(this) << "): " << "Image size: " << srcImg.width << "x" << srcImg.height << "x" << srcImg.channels;
	//orz::Log(orz::INFO) << "QualityAssessment(" << ptr(this) << "): " << "Face info: " << "(" << faceInfo.x << ", " << faceInfo.y << ", " << faceInfo.width << ", " << faceInfo.height << ")";

	if (!impl->pose)
	{
		return 0;
	}

	const int MIN_FACE_WIDTH = 64;

	float face_with = std::sqrt(faceInfo.width * faceInfo.height);
	int satisfied_width = impl->satisfied_face_size;
	float size_score = face_with >= satisfied_width
		? 1
		: (face_with - MIN_FACE_WIDTH) / (impl->satisfied_face_size - MIN_FACE_WIDTH);
	if (size_score < 0) size_score = 0;

	//orz::Log(orz::INFO) << "QualityAssessment(" << ptr(this) << "): " << "evaluate size score: " << size_score;

	float yaw, pitch, roll;
	if (!impl->pose || !impl->pose->Estimate(srcImg, faceInfo, yaw, pitch, roll))
	{
		return size_score;
	}
	float yaw_score = (90 - fabs(yaw) - 30) / 60;
	float pitch_score = (90 - fabs(pitch) - 45) / 45;
	if (yaw_score < 0) yaw_score = 0;
	if (pitch_score < 0) pitch_score = 0;
	float pose_score = 0.5 * yaw_score + 0.5 * pitch_score;

	//orz::Log(orz::INFO) << "QualityAssessment(" << ptr(this) << "): " << "evaluate pose score: " << pose_score;

	float final_score = 0.25 * size_score + 0.75 * pose_score;

	//orz::Log(orz::INFO) << "QualityAssessment(" << ptr(this) << "): " << "evaluate final score: " << final_score;

	float clarity = ClarityEstimate(srcImg, faceInfo);

	if (clarity < impl->clarity_threshold) final_score *= clarity;

	return final_score;
}

static float ReBlur(const unsigned char *data, int width, int height)
{
	float blur_val = 0.0;
	float kernel[9] = { 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0 };
	float *BVer = new float[width * height];//垂直方向低通滤波后的结果
	float *BHor = new float[width * height];//水平方向低通滤波后的结果

	float filter_data = 0.0;
	for (int i = 0; i < height; ++i)//均值滤波
	{
		for (int j = 0; j < width; ++j)
		{
			if (i < 4 || i > height - 5)
			{//处理边界 直接赋值原数据
				BVer[i * width + j] = data[i * width + j];
			}
			else
			{
				filter_data = kernel[0] * data[(i - 4) * width + j] + kernel[1] * data[(i - 3) * width + j] + kernel[2] * data[(i - 2) * width + j] +
					kernel[3] * data[(i - 1) * width + j] + kernel[4] * data[(i)* width + j] + kernel[5] * data[(i + 1) * width + j] +
					kernel[6] * data[(i + 2) * width + j] + kernel[7] * data[(i + 3) * width + j] + kernel[8] * data[(i + 4) * width + j];
				BVer[i * width + j] = filter_data;
			}

			if (j < 4 || j > width - 5)
			{
				BHor[i * width + j] = data[i * width + j];
			}
			else
			{
				filter_data = kernel[0] * data[i * width + (j - 4)] + kernel[1] * data[i * width + (j - 3)] + kernel[2] * data[i * width + (j - 2)] +
					kernel[3] * data[i * width + (j - 1)] + kernel[4] * data[i * width + j] + kernel[5] * data[i * width + (j + 1)] +
					kernel[6] * data[i * width + (j + 2)] + kernel[7] * data[i * width + (j + 3)] + kernel[8] * data[i * width + (j + 4)];
				BHor[i * width + j] = filter_data;
			}

		}
	}

	float D_Fver = 0.0;
	float D_FHor = 0.0;
	float D_BVer = 0.0;
	float D_BHor = 0.0;
	float s_FVer = 0.0;//原始图像数据的垂直差分总和 对应论文中的 s_Fver
	float s_FHor = 0.0;//原始图像数据的水平差分总和 对应论文中的 s_Fhor
	float s_Vver = 0.0;//模糊图像数据的垂直差分总和 s_Vver
	float s_VHor = 0.0;//模糊图像数据的水平差分总和 s_VHor
	for (int i = 1; i < height; ++i)
	{
		for (int j = 1; j < width; ++j)
		{
			D_Fver = std::abs((float)data[i * width + j] - (float)data[(i - 1) * width + j]);
			s_FVer += D_Fver;
			D_BVer = std::abs((float)BVer[i * width + j] - (float)BVer[(i - 1) * width + j]);
			s_Vver += std::max((float)0.0, D_Fver - D_BVer);

			D_FHor = std::abs((float)data[i * width + j] - (float)data[i * width + (j - 1)]);
			s_FHor += D_FHor;
			D_BHor = std::abs((float)BHor[i * width + j] - (float)BHor[i * width + (j - 1)]);
			s_VHor += std::max((float)0.0, D_FHor - D_BHor);
		}
	}
	float b_FVer = (s_FVer - s_Vver) / s_FVer;
	float b_FHor = (s_FHor - s_VHor) / s_FHor;
	blur_val = std::max(b_FVer, b_FHor);

	delete[] BVer;
	delete[] BHor;

	return blur_val;
}


float VIPLQualityAssessment::ClarityEstimate(const VIPLImageData &image, const VIPLFaceInfo &info)
{
	if (!image.data || info.width < 9 || info.height < 9) return 0.0;
	seeta::Image color_data(image.data, image.width, image.height, image.channels);
	seeta::Image gray_data = seeta::gray(color_data);

	seeta::Image src_data = seeta::crop(gray_data, seeta::Rect(info.x, info.y, info.width, info.height));
	float blur_val = ReBlur(src_data.data(), src_data.width(), src_data.height());
	float clarity = 1.0 - blur_val;

	float T1 = 0.3;
	float T2 = 0.55;
	if (clarity <= T1)
	{
		clarity = 0.0;
	}
	else if (clarity >= T2)
	{
		clarity = 1.0;
	}
	else
	{
		clarity = (clarity - T1) / (T2 - T1);
	}

	return clarity;
}
