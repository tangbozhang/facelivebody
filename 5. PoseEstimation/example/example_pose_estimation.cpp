#include <iostream>

#include <VIPLFaceDetector.h>
#include <VIPLPoseEstimation.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


int main()
{
	// 0.1 加载人脸检测模型
	VIPLFaceDetector FD("model/VIPLFaceDetector5.1.0.2kx2k.dat");
	FD.SetMinFaceSize(60);
	FD.SetScoreThresh(0.7, 0.7, 0.85);
	FD.SetVideoStable(true);
	// 0.2 加载姿态估计模型
	VIPLPoseEstimation PE("model/VIPLPoseEstimation1.1.0.ext.dat");

	std::string title = "Pose Esimation";
	cv::namedWindow(title, cv::WINDOW_NORMAL);

	cv::VideoCapture vc(0);
	cv::Mat frame;
	cv::Mat canvas;

	float pre_roll = -180;
	float pre_pitch = -180;
	float pre_yaw = -180;

	while (vc.isOpened())
	{
		if (cv::waitKey(33) >= 0) break;

		vc >> frame;
		if (!frame.data) continue;
		cv::flip(frame, canvas, 1);

		// 1.1 准备人脸检测使用的数据
		VIPLImageData image(frame.cols, frame.rows, frame.channels());
		image.data = frame.data;

		// 1.2 进行人脸检测
		auto infos = FD.Detect(image);

		int line_width = 4;

		std::ostringstream oss;
		for (auto &info : infos)
		{
			float scale = info.width / 300.0;
			float line = 30;
			cv::rectangle(canvas, cv::Rect(canvas.cols - info.x - info.width, info.y, info.width, info.height), cv::Scalar(128, 0, 0), scale * line_width);
			float yaw, pitch, roll;
			// 2.1 进行头部姿态估计，分别有三个方向，以角度表示
			PE.Estimate(image, info, yaw, pitch, roll);

			if (fabs(yaw - pre_yaw) < 1) yaw = pre_yaw;
			if (fabs(pitch - pre_pitch) < 1) pitch = pre_pitch;
			if (fabs(roll - pre_roll) < 1) roll = pre_roll;
			pre_yaw = yaw;
			pre_pitch = pitch;
			pre_roll = roll;

			oss.str("");
			oss << "roll:  " << (roll >= 0 ? " " : "") << roll;
			cv::putText(canvas, oss.str(), cv::Point(canvas.cols - info.x - info.width, info.y - 10 * scale), 0, scale, cv::Scalar(0, 128, 0), scale * line_width);
			oss.str("");
			oss << "pitch: " << (pitch >= 0 ? " " : "") << pitch;
			cv::putText(canvas, oss.str(), cv::Point(canvas.cols - info.x - info.width, info.y - 10 * scale - 1 * scale * line), 0, scale, cv::Scalar(0, 128, 0), scale * line_width);
			oss.str("");
			oss << "yaw:   " << (yaw >= 0 ? " " : "") << yaw;
			cv::putText(canvas, oss.str(), cv::Point(canvas.cols - info.x - info.width, info.y - 10 * scale - 2 * scale * line), 0, scale, cv::Scalar(0, 128, 0), scale * line_width);
		}

		cv::imshow(title, canvas);
	}

	return 0;
}