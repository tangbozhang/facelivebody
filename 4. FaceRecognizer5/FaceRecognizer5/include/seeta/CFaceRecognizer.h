#pragma once

#include "CStruct.h"

#ifdef __cplusplus
extern "C" {
#endif

struct SeetaFaceRecognizer;
struct SeetaFaceRecognizer_SharedModel;

SEETA_C_API struct SeetaFaceRecognizer *SeetaNewFaceRecognizer(struct SeetaModelSetting setting);
SEETA_C_API void SeetaDeleteFaceRecognizer(const struct SeetaFaceRecognizer *obj);

SEETA_C_API int SeetaFaceRecognizer_SetLogLevel(int level);

SEETA_C_API void SeetaFaceRecognizer_SetSingleCalculationThreads(int num);

SEETA_C_API const SeetaFaceRecognizer_SharedModel *SeetaFaceRecognizer_LoadModel(struct SeetaModelSetting setting);
SEETA_C_API void SeetaFaceRecognizer_FreeModel(const struct SeetaFaceRecognizer_SharedModel *model);

SEETA_C_API int SeetaFaceRecognizer_GetCropFaceWidth(const struct SeetaFaceRecognizer *obj);
SEETA_C_API int SeetaFaceRecognizer_GetCropFaceHeight(const struct SeetaFaceRecognizer *obj);
SEETA_C_API int SeetaFaceRecognizer_GetCropFaceChannels(const struct SeetaFaceRecognizer *obj);

SEETA_C_API int SeetaFaceRecognizer_GetExtractFeatureSize(const struct SeetaFaceRecognizer *obj);

SEETA_C_API int SeetaFaceRecognizer_CropFace(const struct SeetaFaceRecognizer *obj, const SeetaImageData image, const SeetaPointF *points, SeetaImageData *face);

SEETA_C_API int SeetaFaceRecognizer_ExtractCroppedFace(const struct SeetaFaceRecognizer *obj, const SeetaImageData image, float *features);

SEETA_C_API int SeetaFaceRecognizer_Extract(const struct SeetaFaceRecognizer *obj, const SeetaImageData image, const SeetaPointF *points, float *features);

SEETA_C_API float SeetaFaceRecognizer_CalculateSimilarity(const struct SeetaFaceRecognizer *obj, const float *features1, const float *features2);

#ifdef __cplusplus
}
#endif