#pragma once
#include "BaseAction.h"
#include <memory>

#include "VIPLEyeBlinkDetector.h"
#include <vector>

class CommonEyeStatus
{
public:
    int status;
};

class CommonEyeBlinkDetector : public CommonValue<CommonEyeStatus>
{
public:
    CommonEyeBlinkDetector(const std::shared_ptr<VIPLEyeBlinkDetector>& model) : bind_model(model)
    {
        bind_image.data = nullptr;
    };

    void bind(const VIPLImageData &image, const VIPLPoint *points);
    CommonEyeStatus update() const override;

    int Detect(const VIPLImageData &image, const VIPLPoint *points);

    bool ClosedEyes(const VIPLImageData &image, const VIPLPoint *points);

private:
    std::shared_ptr<VIPLEyeBlinkDetector> bind_model;
    VIPLImageData bind_image;
    std::vector<VIPLPoint> bind_points;
};

class EyeAction : public BaseAction
{
public:
	using self = EyeAction;

	EyeAction(const std::string &model_path);
    EyeAction(const std::shared_ptr<CommonEyeBlinkDetector> &model);
	~EyeAction();

	void reset() override;
	bool detect(const VIPLImageData& img, const VIPLFaceInfo& info, const VIPLPoint* points5) override;

private:
    std::shared_ptr<CommonEyeBlinkDetector> core;

	int pre_state;
};

