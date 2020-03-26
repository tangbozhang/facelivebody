#include "ShakeHead.h"
#include <algorithm>
#include <iostream>

void CommonPoseEstimation::bind(const VIPLImageData& src_img, const VIPLFaceInfo& info)
{
    bind_image = src_img;
    bind_face = info;
    clear();
}

CommonPose CommonPoseEstimation::update() const
{
    Result result;
    bind_model->Estimate(bind_image, bind_face, result.yaw, result.pitch, result.roll);
    return result;
}

bool CommonPoseEstimation::Estimate(const VIPLImageData& src_img, const VIPLFaceInfo& info, float& yaw, float& pitch,
    float& roll)
{
    if (src_img.data != bind_image.data ||
        bind_face.x != info.x || bind_face.y != info.y ||
        bind_face.width != info.width || bind_face.height != info.height)
    {
        bind(src_img, info);
    }
    Result result = get();
    yaw = result.yaw;
    pitch = result.pitch;
    roll = result.roll;
    return true;
}

ClassShakeHead::ClassShakeHead(const std::string ModelPath,const int shakeHeadAngleThreshold)
{
    auto p = std::make_shared<VIPLPoseEstimation>(ModelPath.c_str(), VIPLPoseEstimation::CPU);
    poseEor = std::make_shared<CommonPoseEstimation>(p);
	frameNum = 0;
	ClassShakeHeadAngleThreshold = shakeHeadAngleThreshold;
}

ClassShakeHead::ClassShakeHead(const std::shared_ptr<VIPLPoseEstimation>& model, const int shakeHeadAngleThreshold)
{
    poseEor = std::make_shared<CommonPoseEstimation>(model);
	frameNum = 0;
	ClassShakeHeadAngleThreshold = shakeHeadAngleThreshold;
}

ClassShakeHead::ClassShakeHead(const std::shared_ptr<CommonPoseEstimation>& model, const int shakeHeadAngleThreshold)
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
