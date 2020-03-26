#pragma once
#include <string>
#include <SeetaStruct.h>
#include "VIPLStruct.h"
#include "seeta/CommonStruct.h"

class SEETAPassiveFaceAntiSpoofing
{
public:
	SEETAPassiveFaceAntiSpoofing(const std::string ModelPath);
	~SEETAPassiveFaceAntiSpoofing();
	double matToTheSeetaNet(const VIPLImageData &ViplImage, const VIPLFaceInfo &face, const VIPLPoint* points5);
private:
	SEETAPassiveFaceAntiSpoofing(const SEETAPassiveFaceAntiSpoofing &other) = delete;
	const SEETAPassiveFaceAntiSpoofing &operator=(const SEETAPassiveFaceAntiSpoofing &other) = delete;

	SeetaCNN_Net* net2timesFace;
	const char* output_blob_name2timesFace;
};
