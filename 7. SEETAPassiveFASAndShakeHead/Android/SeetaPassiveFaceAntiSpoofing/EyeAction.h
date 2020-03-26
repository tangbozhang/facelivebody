#pragma once
#include "BaseAction.h"
#include <memory>

#include "VIPLEyeBlinkDetector.h"

class EyeAction : public BaseAction
{
public:
	using self = EyeAction;

	EyeAction(const std::string &model_path);
	~EyeAction();

	void reset() override;
	bool detect(const VIPLImageData& img, const VIPLFaceInfo& info, const VIPLPoint* points5) override;

private:
	std::shared_ptr<VIPLEyeBlinkDetector> core;

	int pre_state;
};

