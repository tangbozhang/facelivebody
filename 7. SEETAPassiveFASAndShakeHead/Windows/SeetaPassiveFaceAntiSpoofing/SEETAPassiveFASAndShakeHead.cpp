#include "SEETAPassiveFASAndShakeHead.h"

#include "FusionFASProcess.h"

#ifdef NEED_CHECK
#include "encryption/code/checkit.h"
#endif

SEETAPassiveFASAndShakeHead::SEETAPassiveFASAndShakeHead(const std::string ModelPath,
	const int shakeHeadAngleThreshold,
	const int nodHeadAngleThreshold,
	const double clarityTheshold,
	const double fuseThreshold,
	SystemState &systemState,
	const int firstPhaseFrameNum,
	const int detectFrameNum) :impl(new FusionFASProcess(ModelPath, shakeHeadAngleThreshold, nodHeadAngleThreshold, clarityTheshold, fuseThreshold, systemState, firstPhaseFrameNum, detectFrameNum))
{
#ifdef NEED_CHECK
	checkit();
#endif
}

SEETAPassiveFASAndShakeHead::SEETAPassiveFASAndShakeHead(const ModelSetting& model_setting,
    const int shakeHeadAngleThreshold, const int nodHeadAngleThreshold, const double clarityTheshold,
    const double fuseThreshold, SystemState& systemState, const int firstPhaseFrameNum, const int detectFrameNum)
    :impl(new FusionFASProcess(model_setting, shakeHeadAngleThreshold, nodHeadAngleThreshold, clarityTheshold, fuseThreshold, systemState, firstPhaseFrameNum, detectFrameNum))
{
#ifdef NEED_CHECK
    checkit();
#endif
}


SEETAPassiveFASAndShakeHead::~SEETAPassiveFASAndShakeHead()
{

}

void SEETAPassiveFASAndShakeHead::detecion(VIPLImageData VIPLImage, const VIPLFaceInfo& info,
	const VIPLPoint* points5, SystemState& systemState)
{
	impl->detecion(VIPLImage, info, points5, systemState);
}

void SEETAPassiveFASAndShakeHead::reset(SystemState &systemState)
{
	impl->reset(systemState);
}

bool SEETAPassiveFASAndShakeHead::set_actions(const std::vector<SystemState>& actions)
{
	return impl->set_actions(actions);
}

void SEETAPassiveFASAndShakeHead::getLog(double &value1, double &value2, double &yawAngle, double &pitchAngle)
{
	impl->getLog(value1, value2, yawAngle,pitchAngle);
}

VIPLImageData SEETAPassiveFASAndShakeHead::getSnapshot()
{
    return impl->getSnapshot();
}

VIPLImageData SEETAPassiveFASAndShakeHead::getSnapshot(VIPLFaceInfo &face)
{
    return impl->getSnapshot(face);
}