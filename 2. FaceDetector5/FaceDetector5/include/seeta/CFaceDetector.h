#pragma once

#include "CStruct.h"
#include "CFaceInfo.h"

#ifdef __cplusplus
extern "C" {
#endif

struct SeetaFaceDetector;

SEETA_C_API struct SeetaFaceDetector *SeetaNewFaceDetector(struct SeetaModelSetting setting);
SEETA_C_API struct SeetaFaceDetector *SeetaNewFaceDetectorWithCoreSize(struct SeetaModelSetting setting, const SeetaSize core_size);
SEETA_C_API void SeetaDeleteFaceDetector(const struct SeetaFaceDetector *obj);

SEETA_C_API int SeetaFaceDetector_SetLogLevel(int level);

SEETA_C_API void SeetaFaceDetector_SetSingleCalculationThreads(int num);

SEETA_C_API struct SeetaFaceInfoArray SeetaFaceDetector_Detect(struct SeetaFaceDetector *obj, const SeetaImageData image);
SEETA_C_API void SeetaFaceDetector_SetMinFaceSize(struct SeetaFaceDetector *obj, int size);
SEETA_C_API void SeetaFaceDetector_SetImagePyramidScaleFactor(struct SeetaFaceDetector *obj, float factor);
SEETA_C_API void SeetaFaceDetector_SetScoreThresh(struct SeetaFaceDetector *obj, float thresh1, float thresh2, float thresh3);
SEETA_C_API void SeetaFaceDetector_SetVideoStable(struct SeetaFaceDetector *obj, int stable);
SEETA_C_API int SeetaFaceDetector_GetVideoStable(struct SeetaFaceDetector *obj);

SEETA_C_API int SeetaFaceDetector_GetMinFaceSize(struct SeetaFaceDetector *obj);
SEETA_C_API float SeetaFaceDetector_GetImagePyramidScaleFactor(struct SeetaFaceDetector *obj);
SEETA_C_API void SeetaFaceDetector_GetScoreThresh(struct SeetaFaceDetector *obj, float *thresh1, float *thresh2, float *thresh3);

#ifdef __cplusplus
}
#endif