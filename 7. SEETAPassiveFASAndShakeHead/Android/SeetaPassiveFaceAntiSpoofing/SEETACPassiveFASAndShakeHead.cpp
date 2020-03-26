#include "SEETACPassiveFASAndShakeHead.h"
#include "SEETAPassiveFASAndShakeHead.h"

static SystemState SeetaCSystemState2SystemState(SeetaCSystemState state)
{
	switch (state)
	{
	case NO_FACE: return noFace;
	case DETECTING: return detecting;
	case PLEASE_TURN: return pleaseTrun;
	case PASSED: return passed;
	case NOT_PASSED: return notPass;
	case PLEASE_BLINK: return pleaseBlink;
	case PLEASE_NOD_HEAD: return pleaseNodHead;
	default: return noFace;
	}
}

static SeetaCSystemState SystemState2SeetaCSystemState(SystemState state)
{
	switch (state)
	{
	case noFace: return NO_FACE;
	case detecting: return DETECTING;
	case pleaseTrun: return PLEASE_TURN;
	case passed: return PASSED;
	case notPass: return NOT_PASSED;
	case pleaseBlink: return PLEASE_BLINK;
	case pleaseNodHead: return PLEASE_NOD_HEAD;
	default: return NO_FACE;
	}
}

SEETACPassiveFASAndShakeHead* SEETACPassiveFASAndShakeHead_New(const char* ModelPath, const int shakeHeadAngleThreshold,
	const int nodHeadAngleThreshold, const double clarityTheshold, const double fuseThreshold,
	SeetaCSystemState* systemState, const int firstPhaseFrameNum, const int detectFrameNum)
{
	try
	{
		auto local_state = SeetaCSystemState2SystemState(*systemState);
		auto *object = reinterpret_cast<SEETACPassiveFASAndShakeHead *>(
			new SEETAPassiveFASAndShakeHead(ModelPath, shakeHeadAngleThreshold, nodHeadAngleThreshold, clarityTheshold, fuseThreshold, local_state, firstPhaseFrameNum, detectFrameNum));
		*systemState = SystemState2SeetaCSystemState(local_state);
		return object;
	}
	catch (...)
	{
		return nullptr;
	}
}

void SEETACPassiveFASAndShakeHead_Delete(SEETACPassiveFASAndShakeHead* object)
{

	if (object == nullptr) return;
	const auto impl = reinterpret_cast<const SEETAPassiveFASAndShakeHead *>(object);
	delete impl;
}

void SEETACPassiveFASAndShakeHead_detection(SEETACPassiveFASAndShakeHead* object, const VIPLImageData* VIPLImage,
	const VIPLFaceInfo* info, const VIPLPoint* points5, SeetaCSystemState* systemState)
{
	if (object == nullptr) return;
	auto impl = reinterpret_cast<SEETAPassiveFASAndShakeHead *>(object);
	auto local_state = SeetaCSystemState2SystemState(*systemState);

	impl->detecion(*VIPLImage, *info, points5, local_state);

	*systemState = SystemState2SeetaCSystemState(local_state);
}

void SEETACPassiveFASAndShakeHead_reset(SEETACPassiveFASAndShakeHead* object, SeetaCSystemState* systemState)
{
	if (object == nullptr) return;
	auto impl = reinterpret_cast<SEETAPassiveFASAndShakeHead *>(object);
	auto local_state = SeetaCSystemState2SystemState(*systemState);

	impl->reset(local_state);

	*systemState = SystemState2SeetaCSystemState(local_state);
}

int SEETACPassiveFASAndShakeHead_set_actions(SEETACPassiveFASAndShakeHead* object, const SeetaCSystemState* actions,
	int actions_len)
{
	if (object == nullptr) return 0;
	auto impl = reinterpret_cast<SEETAPassiveFASAndShakeHead *>(object);
	std::vector<SystemState> local_actions(actions_len);
	for (int i = 0; i < actions_len; ++i) local_actions[i] = SeetaCSystemState2SystemState(actions[i]);

	return impl->set_actions(local_actions);
}

void SEETACPassiveFASAndShakeHead_getLog(SEETACPassiveFASAndShakeHead* object, double* value1, double* value2,
	double* yawAngle, double* pitchAngle)
{
	if (object == nullptr) return;
	auto impl = reinterpret_cast<SEETAPassiveFASAndShakeHead *>(object);

	impl->getLog(*value1, *value2, *yawAngle, *pitchAngle);
}

void SEETACPassiveFASAndShakeHead_getSnapshot(SEETACPassiveFASAndShakeHead* object, VIPLImageData* snapshot)
{
	if (object == nullptr) return;
	auto impl = reinterpret_cast<SEETAPassiveFASAndShakeHead *>(object);

	*snapshot = impl->getSnapshot();
}
