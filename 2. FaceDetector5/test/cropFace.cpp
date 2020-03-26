#include <iostream>
#include <fstream>
#include <chrono> 

#include <VIPLFaceDetector.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

inline VIPLImageData vipl_convert(const cv::Mat &img)
{
	VIPLImageData vimg(img.cols, img.rows, img.channels());
	vimg.data = img.data;
	return vimg;
}

int main1()
{
	// 0.0 加载模型
	std::cout << "== Load model ==" << std::endl;

	VIPLFaceDetector detector("D:\\Seetatech_Git\\model\\VIPLFaceDetector5.1.2.NIR.640x480.sta", VIPLFaceDetector::AUTO);
	detector.SetMinFaceSize(40);

	std::cout << "Load model success." << std::endl;
	std::cout << std::endl;

	// 0.1 加载待识别图片
	std::cout << "== Load image ==" << std::endl;

	cv::Mat mat = cv::imread("1.jpg", cv::IMREAD_COLOR);
	VIPLImageData image = vipl_convert(mat);

	std::cout << "Load image: " << image.width << "x" << image.height << std::endl;
	std::cout << std::endl;

	// 0.2 人脸检测
	std::cout << "== Start test ==" << std::endl;
	std::vector<VIPLFaceInfo> faces;
	std::cout << std::endl;

	std::cin.get();
	return 0;
}
