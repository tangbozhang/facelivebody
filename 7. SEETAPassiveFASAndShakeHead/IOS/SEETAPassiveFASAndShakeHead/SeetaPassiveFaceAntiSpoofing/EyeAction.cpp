#include "EyeAction.h"
#include <cmath>
#include <iostream>

void CommonEyeBlinkDetector::bind(const VIPLImageData& image, const VIPLPoint* points)
{
    bind_image = image;
    bind_points.clear();
    bind_points.insert(bind_points.end(), points, points + 5);
    clear();
}

CommonEyeStatus CommonEyeBlinkDetector::update() const
{
    Result result;
    result.status = bind_model->Detect(bind_image, bind_points.data());
    std::cout << "Eye status: " << result.status << std::endl;
    return result;
}

bool closed_points(const VIPLPoint &a, const VIPLPoint &b)
{
    const auto dest_x = std::fabs(a.x - b.x);
    if (dest_x > 0.001) return false;
    const auto dest_y = std::fabs(a.y - b.y);
    if (dest_y > 0.001) return false;
    return true;
}

bool closed_points(const VIPLPoint *a, const VIPLPoint *b, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        if (!closed_points(a[i], b[i])) return false;
    }
    return true;
}

int CommonEyeBlinkDetector::Detect(const VIPLImageData& image, const VIPLPoint* points)
{
    if (image.data != bind_image.data ||
        bind_points.empty() ||
        !closed_points(bind_points.data(), points, 5))
    {
        bind(image, points);
    }
    Result result = get();
    return result.status;
}

bool CommonEyeBlinkDetector::ClosedEyes(const VIPLImageData& image, const VIPLPoint* points)
{
    return Detect(image, points) == 3;
}

EyeAction::EyeAction(const std::string &model_path)
{
    auto p = std::make_shared<VIPLEyeBlinkDetector>(model_path.c_str());
    self::core.reset(new CommonEyeBlinkDetector(p));
	pre_state = -1;
}

EyeAction::EyeAction(const std::shared_ptr<CommonEyeBlinkDetector>& model)
{
    core = model;
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
