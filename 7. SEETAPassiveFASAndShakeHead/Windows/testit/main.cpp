#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

//#include <opencv2/opencv.hpp>

#include <VIPLFaceDetector.h>
#include <VIPLPointDetector.h>
#include <VIPLFaceRecognizer.h>

#include"SEETAPassiveFASAndShakeHead.h"

#include "puttextzh.h"
//#include <windows.h> 

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

//人脸库
std::map<int64_t, std::shared_ptr<float>> m_db; // saving face db
int64_t m_max_index = 0;    ///< next saving id

//图像cv转vipl
const VIPLImageData vipl_convert(const cv::Mat &img)
{
	VIPLImageData vimg(img.cols, img.rows, img.channels());
	vimg.data = img.data;
	return vimg;
}
//加载UI图像，并进行固定大小缩放
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
	cv::resize(boxMat, boxMat, imgSize);	//resize视频框
	cv::resize(passMat, passMat, imgSize);
	cv::resize(notPassMat, notPassMat, imgSize);
	cv::resize(pleaseTrunedMat, pleaseTrunedMat, imgSize);
	cv::resize(pleaseBlinkMat, pleaseBlinkMat, imgSize);
	cv::resize(pleaseNodMat, pleaseNodMat, imgSize);
	cv::resize(pleaseFaceTheCameraMat, pleaseFaceTheCameraMat, imgSize);
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

class Snapshot
{
public:
    Snapshot() = default;
    Snapshot(const cv::Mat &image, const VIPLFaceInfo &face)
        : image(image), face(face) {}
    Snapshot(const cv::Mat &image, int x, int y, int width, int height, float score)
        : image(image)
    {
        face.x = x;
        face.y = y;
        face.width = width;
        face.height = height;
        face.score = score;
    }

    cv::Mat image;
    VIPLFaceInfo face;
};

static Snapshot GetSnapshot(const Snapshot &snapshot, cv::Rect &cropper, float scalar = 0.5, int ratio_width = 3, int ratio_height = 4)
{
    auto &frame = snapshot.image;
    auto &rect = snapshot.face;

    Snapshot result;

	cropper.x = rect.x;
	cropper.y = rect.y;
	cropper.width = rect.width;
	cropper.height = rect.height;

    int pad_width = rect.width * scalar;
    int pad_height = rect.height * scalar;
    int padded_x = rect.x - pad_width;
    int padded_y = rect.y - pad_height;
    int padded_width = rect.width + 2 * pad_width;
    int padded_height = rect.height + 2 * pad_height;

    int frame_width = frame.cols;
    int frame_height = frame.rows;

    if (padded_x < 0)
    {
        padded_width += padded_x;
        padded_x = 0;
    }
    if (padded_y < 0)
    {
        padded_height += padded_y;
        padded_y = 0;
    }
    if (padded_x + padded_width > frame_width)
    {
        int edge = padded_x + padded_width - frame_width;
        padded_width -= edge;
    }
    if (padded_y + padded_height > frame_height)
    {
        int edge = padded_y + padded_height - frame_height;
        padded_height -= edge;
    }

    auto score = float(padded_width * padded_height) / (rect.width * scalar * rect.height * scalar);

    // force to 4:3
    // int ratio_width = 3;
    // int ratio_height = 4;
    if (ratio_width <= 0 || ratio_height <= 0)
    {
        ratio_width = frame.cols;
        ratio_height = frame.rows;
    }

    int want_width = std::min(frame_width, rect.width + 2 * pad_width);
    int want_height = std::min(frame_height, rect.height + 2 * pad_height);

    int want_x = rect.x - (want_width - rect.width) / 2;
    int want_y = rect.y - (want_height - rect.height) / 2;

    int unit = std::min(want_width / ratio_width, want_height / ratio_height);
    int fixed_width = unit * ratio_width;
    int fixed_height = unit * ratio_height;
    int fixed_x = want_x - (fixed_width - want_width) / 2;
    int fixed_y = want_y - (fixed_height - want_height) / 2;
    // shift rect to remove pad
    fixed_x = std::max(fixed_x, 0);
    fixed_y = std::max(fixed_y, 0);
    fixed_x = std::min(fixed_x + fixed_width, frame_width) - fixed_width;
    fixed_y = std::min(fixed_y + fixed_height, frame_height) - fixed_height;

    if (fixed_width <= 0 || fixed_height <= 0)
    {
        result = Snapshot();
    }
    else
    {
       /* cv::Rect cropper(fixed_x, fixed_y, fixed_width, fixed_height);*/
		
		cropper.x = fixed_x;
		cropper.y = fixed_y;
		cropper.width = fixed_width;
		cropper.height = fixed_height;

        result = Snapshot(frame(cropper).clone(), rect.x - fixed_x, rect.y - fixed_y, rect.width, rect.height, rect.score);
    }

    return result;
}
//人脸注册
int64_t FaceRegister(VIPLImageData &image, VIPLFaceDetector &FaceDetct, VIPLPointDetector &PointDetect, VIPLFaceRecognizer &FaceRecognizer)
{
	auto faces =FaceDetct.Detect(image);
	if (faces.empty()) return -1;
	VIPLPoint5 points;
	PointDetect.DetectLandmarks(image, faces[0], points);
	std::shared_ptr<float> feats(new float[FaceRecognizer.GetFeatureSize()], std::default_delete<float[]>());
	FaceRecognizer.ExtractFeatureWithCropNormalized(image, points, feats.get());	
	auto new_index = m_max_index++;
	m_db.insert(std::make_pair(new_index, feats));

	return new_index;
}
//特征比对
size_t QueryTop(const float *features, size_t N, int64_t *index, float *similarity, VIPLFaceRecognizer &FaceRecognizer)
{
	//unique_read_lock<rwmutex> _read_locker(m_db_mutex);

	std::vector<std::pair<int64_t, float>> result(m_db.size());
	{
		//std::unique_lock<std::mutex> _locker(m_comparation_mutex);
		size_t i = 0;
		for (auto &line : m_db)
		{
			result[i].first = line.first;
			result[i].second = FaceRecognizer.CalcSimilarityNormalized((FaceFeatures)features, line.second.get());			
			i++;
		}
	}

	std::partial_sort(result.begin(), result.begin() + N, result.end(), [](
		const std::pair<int64_t, float> &a, const std::pair<int64_t, float> &b) -> bool
	{
		return a.second > b.second;
	});
	const size_t top_n = std::min(N, result.size());
	for (size_t i = 0; i < top_n; ++i) 
	{
		index[i] = result[i].first;
		similarity[i] = result[i].second;
	}
	return top_n;
}

int main()
{
	const double value1_Threshold = 0.3;		//非主动活体检测第一个阈值
	const double value2_Threshold = 0.04;	//非主动活体检测第二个阈值
	const float shakeHeadAngleThreshold = 40;	//转头通过的角度
	const float nodHeadAngleThreshold = 30;	//转头通过的角度
	const int totalFrameNum = 80;	//该程序一共检测的帧数（达到设定帧数还不通过检测的话，退出程序）
	const int firstPhaseFrameNum = 10;
	std::string ModelPath = "./Models/";
	SystemState state = noFace;
	std::shared_ptr<SEETAPassiveFASAndShakeHead>  processor = std::make_shared<SEETAPassiveFASAndShakeHead>(ModelPath, shakeHeadAngleThreshold, nodHeadAngleThreshold, value1_Threshold, value2_Threshold, state, firstPhaseFrameNum, totalFrameNum);
	
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

	VIPLFaceDetector FaceDetct((ModelPath + "VIPLFaceDetector5.1.2.m9d6.640x480.sta").c_str());//人脸检测的初始化
	FaceDetct.SetMinFaceSize(80);
	FaceDetct.SetScoreThresh(0.7, 0.7, 0.85);
	FaceDetct.SetImagePyramidScaleFactor(1.414);

	VIPLPointDetector PointDetect((ModelPath + "VIPLPointDetector5.0.pts5.dat").c_str());//关键点检测模型初始化
	FaceDetct.SetVideoStable(true);

	//人脸识别
	VIPLFaceRecognizer FaceRecognizer((ModelPath + "VIPLFaceRecognizer5.0.RN50.D.K.1N.sta").c_str(), VIPLFaceRecognizer::AUTO);
	FaceRecognizer.SetNumThreads(4);
	FaceRecognizer.SetMaxBatchGlobal(1);
	FaceRecognizer.SetCoreNumberGlobal(2);

	//加载人脸库
	std::vector<cv::String> Imgfiles;     //存放文件路径  										 
	cv::glob("./ModelImage/*.jpg", Imgfiles, true); //读取文件夹下所有符合要求的文件路径
	size_t nNum = Imgfiles.size();           //读取的符合条件的文件个数
	for (size_t i = 0; i < nNum; ++i)
	{
		//register face into facedatabase
		std::cerr << "Registering... " << Imgfiles[i] << std::endl;
		cv::Mat mat1 = cv::imread(Imgfiles[i], cv::IMREAD_COLOR);  // Bitmap for BGR layout
		auto image1 = vipl_convert(mat1);
		auto id = FaceRegister(image1, FaceDetct, PointDetect, FaceRecognizer);
		std::cerr << "Registered id = " << id << std::endl;
	}
	std::map<int64_t, std::string> GalleryIndexMap;
	for (size_t i = 0; i < nNum; i++)
	{
		// save index and name pair
		std::string strImageName;
		size_t nPos1, nPos2;
		nPos1 = Imgfiles[i].find_last_of('\\');
		nPos2 = Imgfiles[i].find_last_of('.');
		if ((nPos1 != -1) && (nPos2 != -1))
		{
			strImageName = Imgfiles[i].substr(nPos1 + 1, nPos2 - 1 - nPos1 - 1 + 1);
		}
		GalleryIndexMap.insert(std::make_pair(i, strImageName));
	}

	VIPLPoint5 points;	//创建人脸5点坐标
	for (int test_n = 0; test_n < 1000; ++test_n) {
		processor->reset(state);

		cv::Mat frame;
		/*std::string strCapAddress;
		strCapAddress = "rtsp://admin:Tce123456@192.168.3.46:554/h264/ch0/main/av_stream";*/
		VideoCapture capture(0);//打开视频文件
		//VideoCapture capture(strCapAddress);//打开视频文件
		int videoWidth = 640;	//视频的显示的宽
		int videoHeight = 480;	//视频的显示的高

		capture.set(CV_CAP_PROP_FRAME_WIDTH, videoWidth);	//设置读进摄像头图像的长和宽（竖屏需要另外适配）
		capture.set(CV_CAP_PROP_FRAME_HEIGHT, videoHeight);
		if (!capture.isOpened())	//检测是否正常打开:成功打开时，isOpened返回ture  
		cout << "fail to open!" << endl;

		loadImgAndResize(cv::Size(videoWidth, videoHeight));
		int nFrameNo = 0;//帧号
		while (true)
		{
			if (!capture.read(frame))
			{
				cout << "不能从视频中读取帧！" << endl;
				break;
			}
			nFrameNo++;

			flip(frame, frame, 1);	//左右镜像 左右旋转摄像头，使电脑中图像和人的方向一致
			if (frame.channels() == 4){	//如果为4通道则转为3通道的rgb图像
				cv::cvtColor(frame, frame, CV_RGBA2BGR);
			}
		/*	VIPLImageData VIPLImage(frame.cols, frame.rows, frame.channels());
			VIPLImage.data = frame.data;*/
			auto VIPLImage = vipl_convert(frame);
			std::vector<VIPLFaceInfo> faces = FaceDetct.Detect(VIPLImage);

			int maxFaceID = 0;	//最大人脸的序号
			if (viplFindMaxFace(faces,maxFaceID))
			{
				PointDetect.DetectLandmarks(VIPLImage, faces[maxFaceID], points);//特征点

				//识别
				processor->detecion(VIPLImage, faces[maxFaceID], points, state);//动作
				
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

            VIPLFaceInfo location;
            VIPLImageData snapshot = processor->getSnapshot(location);  // 获取原始视频截图，和人脸位置
            auto face = vipl2mat(snapshot);//格式转换
			if (!face.empty())
			{
				cv::Rect cropper;
                auto cropped_face = GetSnapshot(Snapshot(face, location), cropper, 0.5, 3, 4);   // 从人脸区域外扩0.5，后裁剪出4:3的图片

				rectangle(frame, cropper, cv::Scalar(0, 255, 0), 2, 8, 0);//人脸抠图框
				cv::imshow("SeetaFaceAntiSpoofing", frame);
			
				//cv::imshow("Face", cropped_face.image);
			}
			//人脸识别
			if (viplFindMaxFace(faces, maxFaceID))
			{
				int64_t index = -1;
				float similarity = 0;

				std::shared_ptr<float> feats(new float[FaceRecognizer.GetFeatureSize()], std::default_delete<float[]>());
				FaceRecognizer.ExtractFeatureWithCropNormalized(VIPLImage, points, feats.get());
				QueryTop(feats.get(), 1, &index, &similarity, FaceRecognizer);
				if (similarity > 0.6)
				{
					std::string title = std::to_string(similarity).substr(0, 4);
					std::string strName;
					strName = "帧号：" + std::to_string(nFrameNo) +"\n人：" + GalleryIndexMap[index] + "  相似度：" + title;
					const char * cName = strName.c_str();
					putTextZH(frame, cName, cv::Point(faces[maxFaceID].x, faces[maxFaceID].y - 20), cv::Scalar(0, 255, 0), 30, "微软雅黑");//BGR
				}
				cv::imshow("SeetaFaceAntiSpoofing", frame);
			}

			if (cv::waitKey(10) == 27) break;
			if (state == passed || state == notPass)
			{
                cv::destroyWindow("Face");
				break;
			}
		}
		cv::waitKey(10);
	}
}