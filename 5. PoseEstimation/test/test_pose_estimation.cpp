#include <iostream>
#include <fstream>
#include <chrono>

#include <VIPLPoseEstimation.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

VIPLImageData vipl_convert(const cv::Mat &img)
{
	VIPLImageData vimg(img.cols, img.rows, img.channels());
	vimg.data = img.data;
	return vimg;
}

int main()
{
    // 0.0 加载姿态估计模型
	std::cout << "== Load model ==" << std::endl;

	VIPLPoseEstimation PE("model/VIPLPoseEstimation1.1.0.ext.dat", VIPLPoseEstimation::AUTO);

	std::cout << "Load model success." << std::endl;
	std::cout << std::endl;
    
	// 0.1 加载待识别图片
	std::cout << "== Load image ==" << std::endl;

    cv::Mat mat = cv::imread("1.jpg", cv::IMREAD_COLOR);
	VIPLImageData image = vipl_convert(mat);

	std::cout << "Load image: " << image.width << "x" << image.height << std::endl;
	std::cout << std::endl;
    
    // 0.2 加载检测的人脸，在应用中需要动态检测和特征点定位
	std::cout << "== Load face infomation ==" << std::endl;
	
	VIPLFaceInfo info;
	std::ifstream faceInfo("faceInfo.txt");
	faceInfo >> info.x >> info.y >> info.width >> info.height;
	faceInfo.close();

	std::cout << "Detect face at: (" << info.x << ", " << info.y << ", " << info.width << ", " << info.height << ")" << std::endl;
	std::cout << std::endl;

	// 0.3 姿态估计
	std::cout << "== Start test ==" << std::endl;
	float yaw, pitch, roll;

	int N = 1;
	std::cout << "Compute " << N << " times. " << std::endl;

	using namespace std::chrono;
	microseconds duration(0);
	for (int i = 0; i < N; ++i)
	{
		if (i % 10 == 0) std::cout << '.' << std::flush;
		auto start = system_clock::now();
		PE.Estimate(image, info, yaw, pitch, roll);
		auto end = system_clock::now();
		duration += duration_cast<microseconds>(end - start);
	}
	std::cout << std::endl;
	double spent = 1.0 * duration.count() / 1000 / N;

	std::cout << "Average takes " << spent << " ms " << std::endl;
	std::cout << std::endl;

	// 0.4 获取结果
	std::cout << "== Plot result ==" << std::endl;
	std::cout << "Result: (yaw, pitch, roll) = (" << yaw << ", " << pitch << ", " << roll << ")" << std::endl;
	std::cout << std::endl;

	return 0;
}
