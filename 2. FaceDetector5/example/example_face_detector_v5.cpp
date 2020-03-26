#include <VIPLFaceDetector.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <queue>

const VIPLImageData vipl_convert(const cv::Mat &img)
{
	VIPLImageData vimg(img.cols, img.rows, img.channels());
	vimg.data = img.data;
	return vimg;
}

static cv::Scalar blue(255, 0, 0);
static cv::Scalar green(0, 255, 0);
static cv::Scalar red(0, 0, 255);

static void writeVideo(cv::VideoWriter writer, const cv::Mat &frame, int width, int height)
{
	if (writer.isOpened());
	cv::Mat resized_frame = frame;
	if (frame.cols != width || frame.rows != height)
	{
		cv::resize(frame, resized_frame, cv::Size(width, height), 0, 0, CV_INTER_LINEAR);
	}
	writer << resized_frame;
}

int main(int argc, char *argv[])
{
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
		if (video_file_name == "0" || atoi(video_file_name.c_str()) != 0)
		{
			capture.open(atoi(video_file_name.c_str()));
		}
		else
		{
			capture.open(video_file_name);
		}
	}

	std::string output_file;
	if (argc > 2)
	{
		output_file = argv[2];
	}

	VIPLFaceDetector FD("model/VIPLFaceDetector5.1.2.m9d6.640x480.sta", VIPLFaceDetector::GPU);

	FD.SetMinFaceSize(40);
	FD.SetScoreThresh(0.7, 0.7, 0.85);
	FD.SetVideoStable(true);

	std::vector<std::string> filelist = {};
	bool filelist_mode = !filelist.empty();
	bool single_window_show = false;

	cv::Mat mat;

	std::string title = "Face detector demo";

	capture.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 720);

	// capture.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
	// capture.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);

	int VIDEO_WIDTH = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	int VIDEO_HEIGHT = capture.get(CV_CAP_PROP_FRAME_HEIGHT);

	cv::VideoWriter videoWriter;
	if (!output_file.empty()) videoWriter.open(output_file + ".avi", CV_FOURCC('D', 'I', 'V', 'X'), 25, cv::Size(VIDEO_WIDTH, VIDEO_HEIGHT));

	while (capture.isOpened())
	{
		std::string filename;
		if (filelist_mode)
		{
			if (filelist.empty()) break;
			filename = filelist.back();
			filelist.pop_back();
			capture.open(filename);
		}

		capture >> mat;
		if (mat.empty()) continue;
		if (!mat.data) break;

		auto image = vipl_convert(mat);

		auto infos = FD.Detect(image);

		for (auto info : infos)
		{
			auto color = cv::Scalar(0x30, 0x30, 0xFF);
			color = cv::Scalar(0xFF, 0xFF, 0x97);
			cv::rectangle(mat, cv::Rect(info.x, info.y, info.width, info.height), color, 3);
		}

		cv::putText(mat, "Press ESC to exit.", cv::Point(20, 40), cv::FONT_HERSHEY_DUPLEX, 1, red);
		if (filelist_mode && single_window_show)
		{
			cv::namedWindow(filename, CV_WINDOW_AUTOSIZE);
			cv::imshow(filename, mat);
		}
		else
		{
			cv::namedWindow(title, CV_WINDOW_NORMAL);
			cv::imshow(title, mat);
		}
		writeVideo(videoWriter, mat, VIDEO_WIDTH, VIDEO_HEIGHT);

		if (cv::waitKey(1) >= 0) break;
	}

	if (videoWriter.isOpened())
	{
		videoWriter.release();
	}

	if (filelist_mode) cv::waitKey(0);

	return EXIT_SUCCESS;
}
