#pragma once

#include "CStruct.h"

#ifdef __cplusplus
extern "C" {
#endif

struct SeetaPointDetector;

SEETA_C_API struct SeetaPointDetector *SeetaNewPointDetector(struct SeetaModelSetting setting);
SEETA_C_API void SeetaDeletePointDetector(const struct SeetaPointDetector *obj);

SEETA_C_API int SeetaPointDetector_SetLogLevel(int level);

SEETA_C_API int SeetaPointDetector_GetLandmarkNumber(const struct SeetaPointDetector *obj);

/**
 * \brief detect points
 * \param image source image
 * \param face face location
 * \param points array for output points, with size `GetLandmarkNumber()`
 * \return return false if failed
 */
SEETA_C_API int SeetaPointDetector_Detect(const struct SeetaPointDetector *obj, const SeetaImageData image, const SeetaRect face, SeetaPointF *points);

/**
 * \brief detect points
 * \param image source image
 * \param face face location
 * \param points array for output points, with size `GetLandmarkNumber()`
 * \return return false if failed
 */
SEETA_C_API int SeetaPointDetector_DetectMasks(const struct SeetaPointDetector *obj, const SeetaImageData image, const SeetaRect face, SeetaPointF *points, int *masks);

#ifdef __cplusplus
}
#endif