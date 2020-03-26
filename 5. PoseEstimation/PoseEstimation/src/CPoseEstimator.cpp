#include "seeta/CPoseEstimator.h"
#include "seeta/PoseEstimator.h"

struct SeetaPoseEstimator
{
    seeta::PoseEstimator *impl;
};

SeetaPoseEstimator* SeetaNewPoseEstimator(SeetaModelSetting setting)
{
    std::unique_ptr<SeetaPoseEstimator> obj(new SeetaPoseEstimator);
    try
    {
        obj->impl = new seeta::PoseEstimator(setting);
    }
    catch(const std::exception &)
    {
        return nullptr;
    }
    return obj.release();
}

void SeetaDeletePoseEstimator(const SeetaPoseEstimator* obj)
{
    if (!obj) return;
    delete obj->impl;
    delete obj;
}

int SeetaPoseEstimator_SetLogLevel(int level)
{
    return seeta::PoseEstimator::SetLogLevel(level);
}

void SeetaPoseEstimator_SetSingleCalculationThreads(int num)
{
    seeta::PoseEstimator::SetSingleCalculationThreads(num);
}

void SeetaPoseEstimator_Feed(SeetaPoseEstimator* obj, const SeetaImageData image, const SeetaRect face)
{
    if (!obj || !obj->impl) return;
    obj->impl->Feed(image, face);
}

float SeetaPoseEstimator_Get(SeetaPoseEstimator* obj, SeetaPoseEstimator_Axis axis)
{
    if (!obj || !obj->impl) return 0;
    return obj->impl->Get(seeta::PoseEstimator::Axis(axis));
}

void SeetaPoseEstimator_Estimate(SeetaPoseEstimator* obj, const SeetaImageData& image, const SeetaRect& face,
    float* yaw, float* pitch, float* roll)
{
    if (!obj || !obj->impl) return;
    obj->impl->Estimate(image, face, yaw, pitch, roll);
}
