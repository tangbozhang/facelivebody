#pragma once
#include <VIPLStruct.h>
#include "VIPLPoseEstimation.h"
#include <string>
#include "BaseAction.h"
#include "ShakeHead.h"

class ClassNodHead: public BaseAction
{
public:
	ClassNodHead(const std::string ModelPath,const int nodHeadAngleThreshold);
	ClassNodHead(const std::shared_ptr<VIPLPoseEstimation> &model, const int nodHeadAngleThreshold);
	ClassNodHead(const std::shared_ptr<CommonPoseEstimation> &model, const int nodHeadAngleThreshold);
	~ClassNodHead();
	bool nodHeadDetect(const VIPLImageData &src_img, const VIPLFaceInfo &info,double &pitchAngle);
	void reset(const float& pitch);
	
	void reset() override;
	bool detect(const VIPLImageData& img, const VIPLFaceInfo& info, const VIPLPoint* points5) override;
private:
    std::shared_ptr<CommonPoseEstimation> poseEor;
	float minPitchAngle;
	float maxPitchAngle;
	int frameNum = 0;
	float ClassNodHeadAngleThreshold;

};

