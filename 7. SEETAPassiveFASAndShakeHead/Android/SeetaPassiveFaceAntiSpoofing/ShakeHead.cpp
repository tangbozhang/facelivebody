#include "ShakeHead.h"
#include <algorithm>
#include <iostream>

ClassShakeHead::ClassShakeHead(const std::string ModelPath,const int shakeHeadAngleThreshold)
{

	poseEor = std::make_shared<VIPLPoseEstimation>(ModelPath.c_str(), VIPLPoseEstimation::CPU);
	frameNum = 0;
	ClassShakeHeadAngleThreshold = shakeHeadAngleThreshold;
}

ClassShakeHead::ClassShakeHead(const std::shared_ptr<VIPLPoseEstimation>& model, const int shakeHeadAngleThreshold)
{
	poseEor = model;
	frameNum = 0;
	ClassShakeHeadAngleThreshold = shakeHeadAngleThreshold;
}

ClassShakeHead::~ClassShakeHead()
{

}
void ClassShakeHead::reset(const float& yaw)
{
	minYawAngle = yaw;
	maxYawAngle = yaw;
	frameNum = 0;
}

void ClassShakeHead::reset()
{
	reset(0);
}

bool ClassShakeHead::detect(const VIPLImageData& img, const VIPLFaceInfo& info, const VIPLPoint* points5)
{
	double yaw;
	return shakeHeadDetect(img, info, yaw);
}

bool ClassShakeHead::shakeHeadDetect(const VIPLImageData &src_img, const VIPLFaceInfo &info,double &yawAngle)
{
	float yaw, pitch, roll;
	poseEor->Estimate(src_img, info, yaw, pitch, roll);
	yawAngle = yaw;
	if (frameNum==0)
	{
		reset(yaw);
		frameNum++;
	}
	minYawAngle = std::min(minYawAngle, yaw);
	maxYawAngle = std::max(maxYawAngle, yaw);
	if ((maxYawAngle-minYawAngle)> ClassShakeHeadAngleThreshold)
	{

		return true;
	}
	else
	{
		return false;
	}
}
