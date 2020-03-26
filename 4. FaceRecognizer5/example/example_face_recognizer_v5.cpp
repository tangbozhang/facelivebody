#include <VIPLFaceDetector.h>
#include <VIPLPointDetector.h>
#include <VIPLFaceRecognizer.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <memory>
#include <iostream>
#include <string>
#include <fstream>

const VIPLImageData vipl_convert(const cv::Mat &img)
{
	VIPLImageData vimg(img.cols, img.rows, img.channels());
	vimg.data = img.data;
	return vimg;
}

const cv::Mat vipl_convert(const VIPLImageData &img)
{
	cv::Mat cvimg(img.height, img.width, CV_8UC(img.channels), img.data);
	return cvimg;
}

static std::shared_ptr<float> ExtractFeature(VIPLFaceDetector &FD, VIPLPointDetector &PD, VIPLFaceRecognizer &FR, const VIPLImageData &image)
{
	// 进行人脸检测
	std::vector<VIPLFaceInfo> infos = FD.Detect(image);
	if (infos.size() > 0)
	{
		std::shared_ptr<float> feats(new float[FR.GetFeatureSize()], std::default_delete<float[]>());
		int i = 0;
		VIPLPoint5 points;
		PD.DetectLandmarks(image, infos[i], points);
		FR.ExtractFeatureWithCrop(image, points, feats.get());
		return feats;
	}
	return nullptr;
}

static std::shared_ptr<float> ExtractFeature(VIPLFaceDetector &FD, VIPLPointDetector &PD, VIPLFaceRecognizer &FR, const std::string &path)
{
	// 加载图片
	cv::Mat mat = cv::imread(path, cv::IMREAD_COLOR);  // Bitmap for BGR layout
	if (mat.empty())
	{
		std::cerr << "Can not open image " << path << std::endl;
		return nullptr;
	}
	auto image = vipl_convert(mat);

	auto feats = ExtractFeature(FD, PD, FR, image);

	if (feats == nullptr)
	{
		std::cerr << "Can not detect face on " << path << std::endl;
	}

	return feats;
}

static std::shared_ptr<float> ExtractFeature(VIPLFaceDetector &FD, VIPLPointDetector &PD, VIPLFaceRecognizer &FR, const VIPLImageData &image1, const VIPLImageData &image2)
{
	// 进行人脸检测
	std::vector<VIPLFaceInfo> infos1 = FD.Detect(image1);
	if (infos1.empty())
	{
		std::cerr << "Can not detect face on image1" << std::endl;
		return nullptr;
	}
	std::vector<VIPLFaceInfo> infos2 = FD.Detect(image2);
	if (infos2.empty())
	{
		std::cerr << "Can not detect face on image2" << std::endl;
		return nullptr;
	}
	std::shared_ptr<float> feats(new float[2 * FR.GetFeatureSize()], std::default_delete<float[]>());
	std::vector<VIPLPoint> points(10);
	auto points1 = points.data();
	auto points2 = points.data() + 5;
	PD.DetectLandmarks(image1, infos1[0], points1);
	PD.DetectLandmarks(image2, infos2[0], points2);

	FR.ExtractFeatureWithCrop({ image1, image2 }, points, feats.get());
	return feats;
}

static std::shared_ptr<float> ExtractFeature(VIPLFaceDetector &FD, VIPLPointDetector &PD, VIPLFaceRecognizer &FR, const std::string &path1, const std::string &path2)
{
	// 加载图片
	cv::Mat mat1 = cv::imread(path1, cv::IMREAD_COLOR);  // Bitmap for BGR layout
	if (mat1.empty())
	{
		std::cerr << "Can not open image " << path1 << std::endl;
		return nullptr;
	}
	auto image1 = vipl_convert(mat1);
	cv::Mat mat2 = cv::imread(path2, cv::IMREAD_COLOR);  // Bitmap for BGR layout
	if (mat2.empty())
	{
		std::cerr << "Can not open image " << path2 << std::endl;
		return nullptr;
	}
	auto image2 = vipl_convert(mat2);

	return ExtractFeature(FD, PD, FR, image1, image2);
}

void main(int argc, char *argv[])
{
	if (argc < 3)
	{
		std::cout << "Usage: command image1 image2" << std::endl;
		return;
	}

	VIPLFaceRecognizer::SetNumThreads(4);
	VIPLFaceRecognizer::SetMaxBatchGlobal(1);
	VIPLFaceRecognizer::SetCoreNumberGlobal(2);

	VIPLFaceDetector FD("model/VIPLFaceDetector5.1.2.m9d6.640x480.sta", VIPLFaceDetector::CPU);
	VIPLPointDetector PD("model/VIPLPointDetector5.0.pts5.dat");
	VIPLFaceRecognizer FR("model/VIPLFaceRecognizer5.0.RN30.m5d14.ID.sta", VIPLFaceRecognizer::CPU);

	FD.SetMinFaceSize(40);

	auto feats = ExtractFeature(FD, PD, FR, argv[1], argv[2]);

	if (feats != nullptr)
	{
		float similar = FR.CalcSimilarity(feats.get(), feats.get() + FR.GetFeatureSize());
		std::cout << "Similarity: " << similar << std::endl;
	}
}
