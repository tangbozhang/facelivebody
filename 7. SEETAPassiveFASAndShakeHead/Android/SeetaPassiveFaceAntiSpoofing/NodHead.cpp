#include "NodHead.h"
#include <algorithm>
#include <iostream>

ClassNodHead::ClassNodHead(const std::string ModelPath,const int nodHeadAngleThreshold)
{
	poseEor = std::make_shared<VIPLPoseEstimation>(ModelPath.c_str(), VIPLPoseEstimation::CPU);
	frameNum = 0;
	ClassNodHeadAngleThreshold = nodHeadAngleThreshold;
}

ClassNodHead::ClassNodHead(const std::shared_ptr<VIPLPoseEstimation>& model, const int nodHeadAngleThreshold)
{
	poseEor = model;
	frameNum = 0;
	ClassNodHeadAngleThreshold = nodHeadAngleThreshold;
}

ClassNodHead::~ClassNodHead()
{
	
}

void ClassNodHead::reset(const float& pitch)
{
	minPitchAngle = pitch;
	maxPitchAngle = pitch;
	frameNum = 0;
}

void ClassNodHead::reset()
{
	reset(0);
}

bool ClassNodHead::detect(const VIPLImageData& img, const VIPLFaceInfo& info, const VIPLPoint* points5) 
{
	double pitch;
	return nodHeadDetect(img, info, pitch);
}

bool ClassNodHead::nodHeadDetect(const VIPLImageData &src_img, const VIPLFaceInfo &info,double &pitchAngle)
{
	float yaw, pitch, roll;
	poseEor->Estimate(src_img, info, yaw, pitch, roll);
	pitchAngle = pitch;

	if (frameNum==0)
	{
		reset(pitch);
		frameNum++;
	}
	minPitchAngle = std::min(minPitchAngle, pitch);
	maxPitchAngle = std::max(maxPitchAngle, pitch);
	
	if ((maxPitchAngle-minPitchAngle)> ClassNodHeadAngleThreshold)
	{
		return true;
	}
	else
	{
		return false;
	}
}