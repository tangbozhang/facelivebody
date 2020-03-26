#pragma once
#include <string>
#include "SEETAPassiveFaceAntiSpoofing.h"
#include <memory>
#include "ClarityEstimateClass.h"
#include "ShakeHead.h"
#include "EyeAction.h"
#include "NodHead.h"
#include "SEETAPassiveFASAndShakeHead.h"
#include <vector>
#include <queue>
#include <map>
#include <functional>
using VIPLPoint5 = VIPLPoint[5];
class FusionFASProcess
{
public:
	using self = FusionFASProcess;

	FusionFASProcess(const std::string ModelPath, 
		const int shakeHeadAngleThreshold,
		const int nodHeadAngleThreshold,
		const double clarityTheshold,
		const double fuseThreshold,
		SystemState &systemState,
		const int firstPhaseFrameNum,
		const int detectFrameNum);
	~FusionFASProcess();
	void detecion(VIPLImageData VIPLImage, const VIPLFaceInfo &info, const VIPLPoint* points5, SystemState &systemState);
	void reset(SystemState &systemState);
	void getLog(double &value1, double &value2, double &yawAngle, double &pitchAngle);

	bool set_actions(const std::vector<SystemState>& actions);
	
	VIPLImageData getSnapshot();
	void resetSnapshot();
	float evaluateSnapshotPoseScore(const VIPLImageData& srcImg, const VIPLFaceInfo& faceInfo, const float &yawAngle, const float &pitchAngle, const float & rollAngle) const;
	void updateSnapshot(const VIPLImageData &image, float score);
	void FusionFASProcess::updateSnapshot(const VIPLImageData& image, const VIPLFaceInfo& info, const  float &yawAngle, const  float &pitchAngle, const  float & rollAngle, const double &clarity);
	void FusionFASProcess::updateSnapshot_v2(const VIPLImageData& image, const VIPLFaceInfo& info, const  float &yawAngle, const  float &pitchAngle, const  float & rollAngle, const double &clarity);
	
private:
	VIPLImageData snapshot;
	std::shared_ptr<uint8_t> snapshot_data;
	float snapshot_score = 0;
	size_t snapshot_length = 0;
	
	std::shared_ptr<VIPLPoseEstimation> poseEstimation;
	std::shared_ptr<VIPLEyeBlinkDetector> closeEyeDetector;
	std::shared_ptr<SEETAPassiveFaceAntiSpoofing> antiSpoofing;
	std::shared_ptr<ClarityEstimateClass> ClarityEstimator;
	// std::shared_ptr<ClassShakeHead> ShakeHead;
	// std::shared_ptr<VIPLQualityAssessment>  qualityAssessor;
	int thisShakeHeadAngleThreshold;
	double thisClarityTheshold;
	double thisFuseThreshold;
	bool passiveFASIsPass;
	int thisDetectFrameNum;	//检测的总帧数
	int frameNum;
	float yawAngle=-999, pitchAngle=-999, rollAngle=-999;
	double clarity = -999;
	double passiveFASResult = -999;
	// double qualityValue = -999;
	int thisFirstPhaseFrameNum;	//第一阶段的检测帧数
	double firstPhaseTotal_clarity;	//第一阶段清晰度总值
	double firstPhaseTotal_passive;	//第一阶段dl输出总值
	std::vector<double> firstPhaseVector_clarity;
	std::vector<double> firstPhaseVector_passiveFASResult;
	bool firstPhaseIsPass();
	static double seeta_mean(const seeta::Image &img);
	static double lightJudgment(const seeta::Image &img);
	int isStable(const VIPLPoint *points, VIPLPoint *prepoints, int face_width);
	static bool isMiddle(const float &yawAngle, const float &pitchAngle, const float & rollAngle);
	double clarityJudgment(const double &clarity) const;
	VIPLPoint5 prepoints;	//视频中上一帧的五点定位信息
	std::queue<SystemState> action_list;	///< every need action
	std::queue<SystemState> action_list_template;	///< reset to this default actions

	std::vector<SystemState> thisAction_list;
	std::vector<SystemState> thisAction_list_template;
	std::vector<std::shared_ptr<BaseAction>> action;
	std::map<SystemState, std::shared_ptr<BaseAction>> action_detetors;	// a map of action detectors
};

