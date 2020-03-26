#pragma once

#include "VIPLStruct.h"

class BaseAction
{
public:
	virtual ~BaseAction() {}

	/**
	 * \brief reset state, so can do next detection
	 */
	virtual void reset() = 0;

	/**
	 * \brief detect this action if appear
	 * \param img 
	 * \param info 
	 * \param points5 
	 * \return 
	 */
	virtual bool detect(const VIPLImageData &img, const VIPLFaceInfo &info, const VIPLPoint *points5) = 0;
};

