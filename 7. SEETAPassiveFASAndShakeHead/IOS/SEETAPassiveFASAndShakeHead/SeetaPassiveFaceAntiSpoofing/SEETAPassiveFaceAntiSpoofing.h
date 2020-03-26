#pragma once
#include <string>
#include <SeetaStruct.h>
#include "VIPLStruct.h"
#include "seeta/CommonStruct.h"
#include "SEETAPassiveFASAndShakeHead.h"

class SEETAPassiveFaceAntiSpoofing
{
public:
	// SEETAPassiveFaceAntiSpoofing(const std::string ModelPath);
    SEETAPassiveFaceAntiSpoofing(const SEETAPassiveFASAndShakeHead::ModelSetting &model_setting);
    SEETAPassiveFaceAntiSpoofing(const std::string &model_path);
    SEETAPassiveFaceAntiSpoofing(const SEETAPassiveFASAndShakeHead::Buffer &model_buffer);
	~SEETAPassiveFaceAntiSpoofing();
	double matToTheSeetaNet(const VIPLImageData &ViplImage, const VIPLFaceInfo &face, const VIPLPoint* points5);
private:
    void load(const SEETAPassiveFASAndShakeHead::Buffer &buffer);

	SEETAPassiveFaceAntiSpoofing(const SEETAPassiveFaceAntiSpoofing &other) = delete;
	const SEETAPassiveFaceAntiSpoofing &operator=(const SEETAPassiveFaceAntiSpoofing &other) = delete;

	SeetaCNN_Net* net2timesFace;
	const char* output_blob_name2timesFace;
};
