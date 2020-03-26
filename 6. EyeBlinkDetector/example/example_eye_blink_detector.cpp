#include <iostream>
#include <fstream>
#include <chrono> 

#include <VIPLFaceDetector.h>
#include <VIPLPointDetector.h>
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

auto red = CV_RGB(255, 0, 0);
auto green = CV_RGB(0, 255, 0);
auto blue = CV_RGB(0, 0, 255);

class triger
{
public:
	bool signal(bool level)
	{
		bool now_level = !m_pre_level && level;
		m_pre_level = level;
		return now_level;
	}
private:
	bool m_pre_level = false;
};



int main()
{
	// 0.0 加载模型
	std::cout << "== Load model ==" << std::endl;

	VIPLFaceDetector FD("model/VIPLFaceDetector5.1.2.m9d6.640x480.sta", VIPLFaceDetector::AUTO);
	FD.SetVideoStable(true);
	VIPLPointDetector PD("model/VIPLPointDetector5.0.pts5.dat");
	VIPLEyeBlinkDetector EBD("model/VIPLEyeBlinkDetector1.3.99x99.raw.dat", VIPLEyeBlinkDetector::AUTO);

	std::cout << "Load model success." << std::endl;
	std::cout << std::endl;

	// 0.1 加载待识别图片
	std::cout << "== Open camera ==" << std::endl;

	// cv::VideoCapture capture("WIN_20180609_12_31_01_Pro.mp4");
	cv::VideoCapture capture(0);
	cv::Mat frame, canvas;
	std::stringstream oss;

	triger triger_left, triger_right;
	int count_blink_times = 0;

	while (capture.isOpened())
	{
		capture >> frame;
		if (frame.empty()) continue;
		canvas = frame.clone();

		auto vframe = vipl_convert(frame);

		auto faces = FD.Detect(vframe);

		for (auto &info : faces)
		{

			VIPLPoint points[5];
			PD.DetectLandmarks(vframe, info, points);
			int status= EBD.Detect(vframe, points);
			bool blink = triger_left.signal((status & VIPLEyeBlinkDetector::LEFT_EYE) && (status & VIPLEyeBlinkDetector::RIGHT_EYE));
			// bool blink = triger_left.signal(EBD.ClosedEyes(vframe, points));

			oss.str("");
			oss << "(";
			oss << ((status & VIPLEyeBlinkDetector::LEFT_EYE) ? "close" : "open");
			oss << ", ";
			oss << ((status & VIPLEyeBlinkDetector::RIGHT_EYE) ? "close" : "open");
			oss << ")";

			if (blink)
			{
				count_blink_times++;
				std::cout << "Blink " << count_blink_times << std::endl;
			}

			cv::rectangle(canvas, cv::Rect(info.x, info.y, info.width, info.height), cv::Scalar(128, 0, 0), 3);
			cv::putText(canvas, oss.str(), cv::Point(info.x, info.y - 10), 0, 0.5, cv::Scalar(0, 128, 0), 2);
		}

		cv::imshow("Faces", canvas);
		auto key = cv::waitKey(30);
		if (key >= 0) break;
	}

	return EXIT_SUCCESS;
}