#include <iostream>
#include <fstream>
#include <chrono> 

#include <VIPLFaceRecognizer.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <memory>

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

	VIPLFaceRecognizer FR("model/VIPLFaceRecognizer5.0.RN50.49w.s4.1N.ats", VIPLFaceRecognizer::AUTO);
	std::string outFeatFilename = "feats.RN2.txt";

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

	// 0.3 裁剪出单独人脸
	std::cout << "== Crop face ==" << std::endl;

	cv::Mat face(FR.GetCropHeight(), FR.GetCropWidth(), CV_8UC(FR.GetCropChannels()));
	VIPLImageData vface = vipl_convert(face);
	FR.CropFace(image, points, vface);

	std::cout << std::endl;

	// 0.4 进行识别
	std::unique_ptr<float[]> feats(new float[FR.GetFeatureSize()]);
	bool success;

	int N = 1;
	std::cout << "Compute " << N << " times. " << std::endl;

	using namespace std::chrono;
	microseconds duration(0);
	for (int i = 0; i < N; ++i)
	{
		if (i % 10 == 0) std::cout << '.' << std::flush;
		auto start = system_clock::now();
		success = FR.ExtractFeature(vface, feats.get());
		auto end = system_clock::now();
		duration += duration_cast<microseconds>(end - start);
	}
	std::cout << std::endl;
	double spent = 1.0 * duration.count() / 1000 / N;

	std::cout << "Average takes " << spent << " ms " << std::endl;
	std::cout << std::endl;

	// 0.5 获取结果
	std::cout << "== Plot result ==" << std::endl;
	std::ofstream outFeat(outFeatFilename);
	std::cout << "First 10 features of " << FR.GetFeatureSize() << ":" << std::endl;
	for (int i = 0; i < FR.GetFeatureSize(); ++i)
	{
		if (i < 10) std::cout << feats[i] << std::endl;
		outFeat << feats[i] << std::endl;
	}

	outFeat.close();
	cv::imwrite("copped_face.png", face);

	std::cout << std::endl;
	std::cout << "Output file: " << outFeatFilename << std::endl;
	std::cout << "Output file: copped_face.png" << std::endl;
	std::cout << std::endl;

	return 0;
}
