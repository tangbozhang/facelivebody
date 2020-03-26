#pragma once
#include "VIPLStruct.h"

class ClarityEstimateClass
{
public:
	ClarityEstimateClass();
	~ClarityEstimateClass();
	float clarityEstimate(const VIPLImageData &viplImage, const VIPLFaceInfo &info);
private:
	float ReBlur(const unsigned char *data, int width, int height) const;
};

