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

int main()
{
	// 0.0 加载模型
	std::cout << "== Load model ==" << std::endl;

	VIPLFaceDetector detector("model/VIPLFaceDetector5.1.2.m9d6.640x480.sta", VIPLFaceDetector::AUTO);
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

	int N = 1;
	std::cout << "Compute " << N << " times. " << std::endl;

	using namespace std::chrono;
	microseconds duration(0);
	for (int i = 0; i < N; ++i)
	{
		if (i % 10 == 0) std::cout << '.' << std::flush;
		auto start = system_clock::now();
		faces = detector.Detect(image);
		auto end = system_clock::now();
		duration += duration_cast<microseconds>(end - start);
	}
	std::cout << std::endl;
	double spent = 1.0 * duration.count() / 1000 / N;
	
	std::cout << "Average takes " << spent << " ms " << std::endl;
	std::cout << std::endl;

	// 0.4 获取结果
	std::cout << "== Plot result ==" << std::endl;
	std::ofstream faceInfo("faceInfo.txt");
	cv::Rect face_rect;
	int32_t num_face = static_cast<int32_t>(faces.size());
	std::cout << "Detected " << num_face << " face" << (num_face > 1 ? "s" : "") << "." << std::endl;
	std::cout << "Writing faceInfo.txt ..." << std::endl;
	for (int32_t i = 0; i < num_face; i++) {
		face_rect.x = faces[i].x;
		face_rect.y = faces[i].y;
		face_rect.width = faces[i].width;
		face_rect.height = faces[i].height;
		faceInfo << faces[i].x << " " << faces[i].y << " " << faces[i].width << " " << faces[i].height << std::endl;
		cv::rectangle(mat, face_rect, CV_RGB(0, 0, 255), 4, 8, 0);
		std::cout << "Face " << i + 1 << ": (" << faces[i].x << ", " << faces[i].y << ", " << faces[i].width << ", " << faces[i].height << ")" << std::endl;
	}
	faceInfo.close();
	cv::imwrite("image_rect.jpg", mat);

	std::cout << std::endl;
	std::cout << "Output file: faceInfo.txt" << std::endl;
	std::cout << "Output file: image_rect.jpg" << std::endl;
	std::cout << std::endl;

	std::cin.get();
	return 0;
}
