#pragma once
#include <VIPLStruct.h>
#include "VIPLPoseEstimation.h"
#include <string>
#include "BaseAction.h"

class ClassShakeHead : public BaseAction
{
public:
	ClassShakeHead(const std::string ModelPath,const int shakeHeadAngleThreshold);
	ClassShakeHead(const std::shared_ptr<VIPLPoseEstimation>& model, const int shakeHeadAngleThreshold);
	~ClassShakeHead();
	bool shakeHeadDetect(const VIPLImageData &src_img, const VIPLFaceInfo &info,double &yawAngle);
	void reset(const float& yaw);

	void reset() override;
	bool detect(const VIPLImageData& img, const VIPLFaceInfo& info, const VIPLPoint* points5) override;
private:
	std::shared_ptr<VIPLPoseEstimation> poseEor;
	float minYawAngle;
	float maxYawAngle;
	int frameNum = 0;
	float ClassShakeHeadAngleThreshold;
};

