#pragma once
#include <VIPLStruct.h>
#include "VIPLPoseEstimation.h"
#include <string>
#include "BaseAction.h"

class CommonPose
{
public:
    float yaw = 0;
    float pitch = 0;
    float roll = 0;
};

class CommonPoseEstimation : public CommonValue<CommonPose>
{
public:
    CommonPoseEstimation(const std::shared_ptr<VIPLPoseEstimation>& model) : bind_model(model)
    {
        bind_image.data = nullptr;
        bind_face.x = 0;
        bind_face.y = 0;
        bind_face.width = 0;
        bind_face.height = 0;
    };

    void bind(const VIPLImageData &src_img, const VIPLFaceInfo &info);
    CommonPose update() const override;

    bool Estimate(const VIPLImageData &src_img, const VIPLFaceInfo &info, float &yaw, float &pitch, float &roll);

private:
    std::shared_ptr<VIPLPoseEstimation> bind_model;
    VIPLImageData bind_image;
    VIPLFaceInfo bind_face;
};

class ClassShakeHead : public BaseAction
{
public:
	ClassShakeHead(const std::string ModelPath,const int shakeHeadAngleThreshold);
	ClassShakeHead(const std::shared_ptr<VIPLPoseEstimation>& model, const int shakeHeadAngleThreshold);
    ClassShakeHead(const std::shared_ptr<CommonPoseEstimation>& model, const int shakeHeadAngleThreshold);
	~ClassShakeHead();
	bool shakeHeadDetect(const VIPLImageData &src_img, const VIPLFaceInfo &info,double &yawAngle);
	void reset(const float& yaw);

	void reset() override;
	bool detect(const VIPLImageData& img, const VIPLFaceInfo& info, const VIPLPoint* points5) override;
private:
    std::shared_ptr<CommonPoseEstimation> poseEor;
	float minYawAngle;
	float maxYawAngle;
	int frameNum = 0;
	float ClassShakeHeadAngleThreshold;
};

