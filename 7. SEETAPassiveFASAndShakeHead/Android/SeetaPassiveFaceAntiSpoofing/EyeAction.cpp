#include "EyeAction.h"


EyeAction::EyeAction(const std::string &model_path)
{
	self::core.reset(new VIPLEyeBlinkDetector(model_path.c_str()));
	pre_state = -1;
}

EyeAction::~EyeAction()
{
}

void EyeAction::reset()
{
	pre_state = -1;
}

bool EyeAction::detect(const VIPLImageData& img, const VIPLFaceInfo& info, const VIPLPoint* points5)
{
	if (points5 == nullptr) return false;
	auto state = self::core->Detect(img, points5);
	if (pre_state < 0)
	{
		pre_state = state;
		return false;
	}

	bool blink = (pre_state == 0) & (state == (VIPLEyeBlinkDetector::LEFT_EYE | VIPLEyeBlinkDetector::RIGHT_EYE));

	pre_state = state;

	return blink;
}
