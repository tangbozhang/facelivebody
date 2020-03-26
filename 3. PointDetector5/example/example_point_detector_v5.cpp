#include <VIPLFaceDetector.h>
#include <VIPLPointDetector.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include "stable.h"
#include <ctime>
#include <chrono>

const VIPLImageData vipl_convert(const cv::Mat &img)
{
	VIPLImageData vimg(img.cols, img.rows, img.channels());
	vimg.data = img.data;
	return vimg;
}

static cv::Scalar blue(255, 0, 0);
static cv::Scalar green(0, 255, 0);
static cv::Scalar red(0, 0, 255);

int main(int argc, char *argv[])
{
	std::time_t now_c = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	char buf[32];
	strftime(buf, 32, "%Y-%m-%d %H:%M:%S", std::localtime(&now_c));
	std::cout << buf << std::endl;

	cv::VideoCapture capture;
	if (argc < 2)
	{
		std::cout << "Open Camera(0)" << std::endl;
		std::cout << "Use \"" << argv[0] << " video_file_name\" to open video." << std::endl;
		capture.open(0);
	}
	else
	{
		std::string video_file_name = argv[1];
		std::cout << "Open " << video_file_name << std::endl;
		capture.open(video_file_name);
	}

	std::string output_file = "1";
	if (argc > 2)
	{
		output_file = argv[2];
	}

	VIPLFaceDetector FD("model/VIPLFaceDetector5.1.0.dat");
	VIPLPointDetector PD("model/VIPLPointDetector5.0.pts19.mask.dat");

	FD.SetMinFaceSize(120);
	FD.SetScoreThresh(0.7, 0.7, 0.85);
	FD.SetVideoStable(true);
	PD.SetStable(true);


	cv::Mat mat;

	std::vector<PointStable> ps;

	std::string title = "Point detector demo";
	cv::namedWindow(title, CV_WINDOW_NORMAL);

	// capture.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
	// capture.set(CV_CAP_PROP_FRAME_HEIGHT, 720);

	// capture.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
	// capture.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);

	int VIDEO_WIDTH = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	int VIDEO_HEIGHT = capture.get(CV_CAP_PROP_FRAME_HEIGHT);

	int landmark_num = PD.LandmarkNum();

	std::vector<VIPLPoint> landmarks;
	std::vector<int> masks;

	while (capture.isOpened())
	{
		capture >> mat;
		if (!mat.data) break;

		if (ps.empty()) ps.resize(1);

		cv::Mat blur_mat = ps[0].blur_image(mat);

		auto blur_image = vipl_convert(blur_mat);

		auto image = vipl_convert(mat);

		auto infos = FD.Detect(blur_image);

		std::sort(infos.begin(), infos.end(), [](const VIPLFaceInfo &lhs, const VIPLFaceInfo &rhs)
		{
			return lhs.x < rhs.x;
		});

		ps.resize(infos.size());

		for (int n = 0; n < infos.size(); ++n)
		{
			auto &info = infos[n];

            PD.DetectLandmarks(blur_image, info, landmarks, masks);

			landmarks = ps[n].deal_landmarks(infos[n], landmarks);

			// cv::rectangle(mat, cv::Rect(info.x, info.y, info.width, info.height), blue, 3);
			auto color = cv::Scalar(200, 200, 200);
			color = cv::Scalar(0xFF, 0xFF, 0x97);
            for (size_t i = 0; i < landmarks.size(); ++i)
			{
                auto &point = landmarks[i];
                auto &mask = masks[i];
                cv::circle(mat, cv::Point(point.x, point.y), 3, (mask ? red : green), -1);
			}
		}

		cv::imshow(title, mat);

		if (cv::waitKey(1) >= 0) break;
	}

	return EXIT_SUCCESS;
}
