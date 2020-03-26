#include <iostream>
#include <fstream>
#include <chrono> 

#include <VIPLEyeBlinkDetector.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

const VIPLImageData vipl_convert(const cv::Mat &img)
{
	VIPLImageData vimg(img.cols, img.rows, img.channels());
	vimg.data = img.data;
	return vimg;
}

const cv::Mat vipl_convert(const VIPLImageData &vimg)
{
	cv::Mat img(vimg.height, vimg.width, CV_8UC(vimg.channels));
#if _MSC_VER >= 1600
	memcpy_s(img.data, vimg.height * vimg.width * vimg.channels, vimg.data, vimg.height * vimg.width * vimg.channels);
#else
	memcpy(img.data, vimg.data, vimg.height * vimg.width * vimg.channels);
#endif
	return img;
}

#define THREAD_RUN

int main()
{
	// 0.0 加载模型
	std::cout << "== Load model ==" << std::endl;

	VIPLEyeBlinkDetector EBD("model/VIPLEyeBlinkDetector1.2.99x99.raw.dat", VIPLEyeBlinkDetector::AUTO);

	std::cout << "Load model success." << std::endl;
	std::cout << std::endl;

	// 0.1 加载待识别图片
	std::cout << "== Load image ==" << std::endl;

	cv::Mat mat = cv::imread("1.png", cv::IMREAD_COLOR);
	VIPLImageData image = vipl_convert(mat);

	std::cout << "Load image: " << image.width << "x" << image.height << std::endl;
	std::cout << std::endl;

	// 0.2 加载检测的人脸，在应用中需要动态检测和特征点定位
	std::cout << "== Load face infomation ==" << std::endl;

	VIPLPoint points[5];
	std::ifstream landmarks("landmarks.txt");
	std::cout << "Detect landmarks at: [" << std::endl;
	for (int i = 0; i < 5; ++i)
	{
		landmarks >> points[i].x >> points[i].y;
		std::cout << "(" << points[i].x << ", " << points[i].y << ")," << std::endl;
	}
	std::cout << "]" << std::endl;
	landmarks.close();

	std::cout << std::endl;

	// 0.2 眨眼检测
	int status = 0;

	int N = 100;
	std::cout << "Compute " << N << " times. " << std::endl;

	using namespace std::chrono;
	microseconds duration(0);
	for (int i = 0; i < N; ++i)
	{
		if (i % 10 == 0) std::cout << '.' << std::flush;
		auto start = system_clock::now();
		status = EBD.Detect(image, points);
		auto end = system_clock::now();
		duration += duration_cast<microseconds>(end - start);
	}
	std::cout << std::endl;
	double spent = 1.0 * duration.count() / 1000 / N;

	std::cout << "Average takes " << spent << " ms " << std::endl;
	std::cout << std::endl;

	// 0.5 获取结果
	std::cout << "== Plot result ==" << std::endl;
	std::cout << "Eye status: (left, right) = " << "(";
	std::cout << ((status & VIPLEyeBlinkDetector::LEFT_EYE) ? "close" : "open");
	std::cout << ", ";
	std::cout << ((status & VIPLEyeBlinkDetector::RIGHT_EYE) ? "close" : "open");
	std::cout << ")";
	std::cout << std::endl;

	return 0;
}