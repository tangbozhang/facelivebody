#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <VIPLFaceDetector.h>
#include <VIPLPointDetector.h>
#include"SEETAPassiveFASAndShakeHead.h"
using namespace std;
using namespace cv;
using VIPLPoint5 = VIPLPoint[5];
static cv::Mat noface;	//没有人脸
static cv::Mat boxMat;	//开始静默活体检测阶段
static cv::Mat passMat;			//通过时附加的框
static cv::Mat notPassMat;	//未通过时附加的框
static cv::Mat pleaseTrunedMat;	//请转头提示框
static cv::Mat pleaseBlinkMat;	//请眨眼提示框
static cv::Mat pleaseNodMat;	//请点头提示框
static cv::Mat pleaseFaceTheCameraMat;	//请正面朝向摄像头
static cv::Mat verifyingMat;	//请正面朝向摄像头
void loadImgAndResize(const cv::Size imgSize)
{
	noface = cv::imread("ui/noface.png");
	boxMat = cv::imread("ui/box.png");				//活体检测时附加的框
	passMat = cv::imread("ui/pass.png");			//通过时附加的框
	notPassMat = cv::imread("ui/notPass.png");	//未通过时附加的框
	pleaseTrunedMat = cv::imread("ui/pleaseTurn.png");	//未通过时附加的框
	pleaseBlinkMat = cv::imread("ui/pleaseBlink.png");	//
	pleaseNodMat = cv::imread("ui/pleaseNod.png");	//
	pleaseFaceTheCameraMat = cv::imread("ui/pleaseFaceTheCamera.png"); 
    verifyingMat = cv::imread("ui/verifying.png");
	cv::resize(boxMat, boxMat, imgSize);	//resize视频框
	cv::resize(passMat, passMat, imgSize);
	cv::resize(notPassMat, notPassMat, imgSize);
	cv::resize(pleaseTrunedMat, pleaseTrunedMat, imgSize);
	cv::resize(pleaseBlinkMat, pleaseBlinkMat, imgSize);
	cv::resize(pleaseNodMat, pleaseNodMat, imgSize);
	cv::resize(pleaseFaceTheCameraMat, pleaseFaceTheCameraMat, imgSize);
    cv::resize(verifyingMat, verifyingMat, imgSize);
}
/**
* \brief 将摄像头拍摄的照片与ui进行融合
* \param frame 当前画面
* \param systemState 系统当前状态
*/
void addWeightedToImage(cv::Mat& frame, const int &systemState)
{
	switch (systemState)
	{
	case noFace:
	{
		cv::addWeighted(noface, 1.0, frame, 0.65, 35.0, frame);
		break;
	}
	case detecting:
	{
		cv::addWeighted(boxMat, 1.0, frame, 0.65, 35.0, frame);
		break;
	}
	case passed:
	{
		cv::addWeighted(passMat, 1.0, frame, 0.65, 35.0, frame);
		break;
	}
	case notPass:
	{
		cv::addWeighted(notPassMat, 1.0, frame, 0.65, 35.0, frame);
		break;
	}
	case pleaseTrun:
	{
		cv::addWeighted(pleaseTrunedMat, 1.0, frame, 0.65, 35.0, frame);
		break;
	}
	case pleaseBlink:
	{
		cv::addWeighted(pleaseBlinkMat, 1.0, frame, 0.65, 35.0, frame);
		break;
	}
	case pleaseNodHead:
	{
		cv::addWeighted(pleaseNodMat, 1.0, frame, 0.65, 35.0, frame);
		break;
    }
    case pleaseFaceTheCamera:
    {
        cv::addWeighted(pleaseFaceTheCameraMat, 1.0, frame, 0.65, 35.0, frame);
        break;
    }
    case verifying:
    {
        cv::addWeighted(verifyingMat, 1.0, frame, 0.65, 35.0, frame);
        break;
    }
	default:
	{
		cout << "ERROR: system state is false!" << endl;
	}
	}
}
bool viplFindMaxFace(const std::vector<VIPLFaceInfo>& faces, int &maxFaceID)
{
	if (faces.size() > 0)
	{
		maxFaceID = 0;
		double faceMaxWidth = faces[0].width;
		for (int j = 1; j < faces.size(); ++j) {
			if (faceMaxWidth < faces[j].width) {
				faceMaxWidth = faces[j].width;
				maxFaceID = j;
			}
		}
		return true;
	}
	else
	{
		return false;
	}
}


cv::Mat vipl2mat(const VIPLImageData &vimg)
{
	return cv::Mat(vimg.height, vimg.width, CV_8UC(vimg.channels), vimg.data);
}

int main()
{
	const double value1_Threshold = 0.3;		//非主动活体检测第一个阈值
	const double value2_Threshold = 0.04;	//非主动活体检测第二个阈值
	const float shakeHeadAngleThreshold = 40;	//转头通过的角度
	const float nodHeadAngleThreshold = 30;	//转头通过的角度
	const int totalFrameNum = 80;	//该程序一共检测的帧数（达到设定帧数还不通过检测的话，退出程序）
	const int firstPhaseFrameNum = 10;
	std::string ModelPath = "./model/";
    SystemState state = noFace;

    SEETAPassiveFASAndShakeHead::ModelSetting model_setting;
    model_setting.PE = ModelPath + "/VIPLPoseEstimation1.1.0.ext.dat";
    model_setting.EBD = ModelPath + "/VIPLEyeBlinkDetector1.3.99x99.raw.dat";
    model_setting.FAS = ModelPath + "/_acc97_res14YuvAlignment_68000_0622.bin";
    model_setting.FR = ModelPath + "/VIPLFaceAntiSpoofing.light.sta";
    std::shared_ptr<SEETAPassiveFASAndShakeHead>  processor = std::make_shared<SEETAPassiveFASAndShakeHead>(model_setting, shakeHeadAngleThreshold, nodHeadAngleThreshold, value1_Threshold, value2_Threshold, state, firstPhaseFrameNum, totalFrameNum);
	
	// 没有动作，只做活体
	// processor->set_actions({ });
	// 只检测单一动作
	// processor->set_actions({ pleaseBlink });
	// 摇头，眨眼二选一
	// processor->set_actions({ pleaseTrun, pleaseBlink });
	// 摇头，眨眼、点头三选一
	processor->set_actions({ pleaseTrun, pleaseBlink, pleaseNodHead });
	// 默认是做摇头检测
	//processor->set_actions({ pleaseNodHead });
	VIPLFaceDetector FaceDetct((ModelPath + "VIPLFaceDetector5.1.0.640x480.dat").c_str());//人脸检测的初始化
	FaceDetct.SetMinFaceSize(80);
	FaceDetct.SetScoreThresh(0.7, 0.7, 0.85);
	FaceDetct.SetImagePyramidScaleFactor(1.414);
	VIPLPointDetector PointDetect((ModelPath + "VIPLPointDetector5.0.pts5.dat").c_str());//关键点检测模型初始化
	FaceDetct.SetVideoStable(true);
	VIPLPoint5 points;	//创建人脸5点坐标
	for (int test_n = 0; test_n < 1000; ++test_n) {
		processor->reset(state);
		cv::Mat frame;
		VideoCapture capture(0);//打开视频文件
		int videoWidth = 640;	//视频的显示的宽
		int videoHeight = 480;	//视频的显示的高

		capture.set(CV_CAP_PROP_FRAME_WIDTH, videoWidth);	//设置读进摄像头图像的长和宽（竖屏需要另外适配）
		capture.set(CV_CAP_PROP_FRAME_HEIGHT, videoHeight);
		if (!capture.isOpened())	//检测是否正常打开:成功打开时，isOpened返回ture  
		cout << "fail to open!" << endl;
		loadImgAndResize(cv::Size(videoWidth, videoHeight));
		while (true)
		{
			if (!capture.read(frame))
			{
				cout << "不能从视频中读取帧！" << endl;
				break;
			}
			flip(frame, frame, 1);	//左右旋转摄像头，使电脑中图像和人的方向一致
			if (frame.channels() == 4){	//如果为4通道则转为3通道的rgb图像
				cv::cvtColor(frame, frame, CV_RGBA2BGR);
			}
			VIPLImageData VIPLImage(frame.cols, frame.rows, frame.channels());
			VIPLImage.data = frame.data;
			std::vector<VIPLFaceInfo> faces = FaceDetct.Detect(VIPLImage);

			int maxFaceID = 0;	//最大人脸的序号
			if (viplFindMaxFace(faces,maxFaceID))
			{
				PointDetect.DetectLandmarks(VIPLImage, faces[maxFaceID], points);
				processor->detecion(VIPLImage, faces[maxFaceID], points, state);
				
				double value1, value2, yaw, pitch;
				processor->getLog(value1, value2, yaw, pitch);
				std::cout << "value1: " << value1 << ", value2: " << value2 << "yaw: "<<yaw<<" pitch: "<<pitch<<std::endl;

				for (int i = 0; i < 5; ++i)
				{
					cv::circle(frame, cv::Point(points[i].x, points[i].y), 2, cv::Scalar(0, 255, 0), -1);
				}
			}
			else	//没有人脸，重置
			{
				processor->reset(state);
			}
			for (int i = 0; i < faces.size(); i++)
			{
				rectangle(frame, cv::Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height), cv::Scalar(232, 162, 0), 2, 8, 0);//画人脸检测框
			}
			addWeightedToImage(frame, state);
			cv::imshow("SeetaFaceAntiSpoofing", frame);

            VIPLFaceInfo info;
            auto face = vipl2mat(processor->getSnapshot(info));
			if (!face.empty())
            {
                rectangle(face, cv::Rect(info.x, info.y, info.width, info.height), cv::Scalar(232, 162, 0), 2, 8, 0);//画人脸检测框
				cv::imshow("Face", face);
			}
			if (cv::waitKey(30) == 27) break;
			if (state == passed || state == notPass)
			{
				break;
			}
		}
		cv::waitKey(100);
	}
}