#pragma once

#include "CStruct.h"
#include "CStream.h"

#ifdef __cplusplus
extern "C" {
#endif

struct SeetaFaceDatabase;

SEETA_C_API struct SeetaFaceDatabase *SeetaNewFaceDatabase(struct SeetaModelSetting setting);
SEETA_C_API struct SeetaFaceDatabase *SeetaNewFaceDatabase2(struct SeetaModelSetting setting, int extraction_core_number, int comparation_core_number);
SEETA_C_API void SeetaDeleteFaceDatabase(const struct SeetaFaceDatabase *obj);

SEETA_C_API int SeetaFaceDatabase_SetLogLevel(int level);

SEETA_C_API void SeetaFaceDatabase_SetSingleCalculationThreads(int num);

SEETA_C_API int SeetaFaceDatabase_GetCropFaceWidth();
SEETA_C_API int SeetaFaceDatabase_GetCropFaceHeight();
SEETA_C_API int SeetaFaceDatabase_GetCropFaceChannels();

SEETA_C_API bool SeetaFaceDatabase_CropFace(const SeetaImageData image, const SeetaPointF *points, SeetaImageData *face);

SEETA_C_API float SeetaFaceDatabase_Compare(struct SeetaFaceDatabase *obj,
    const SeetaImageData image1, const SeetaPointF *points1,
    const SeetaImageData image2, const SeetaPointF *points2);

SEETA_C_API float SeetaFaceDatabase_CompareByCroppedFace(struct SeetaFaceDatabase *obj,
    const SeetaImageData cropped_face_image1,
    const SeetaImageData cropped_face_image2);

SEETA_C_API int64_t SeetaFaceDatabase_Register(struct SeetaFaceDatabase *obj, const SeetaImageData image, const SeetaPointF *points);
SEETA_C_API int64_t SeetaFaceDatabase_RegisterByCroppedFace(struct SeetaFaceDatabase *obj, const SeetaImageData cropped_face_image);
SEETA_C_API int SeetaFaceDatabase_Delete(struct SeetaFaceDatabase *obj, int64_t index);    // return effected lines, 1 for succeed, 0 for nothing
SEETA_C_API void SeetaFaceDatabase_Clear(struct SeetaFaceDatabase *obj); // clear all faces

SEETA_C_API size_t SeetaFaceDatabase_Count(struct SeetaFaceDatabase *obj);
SEETA_C_API int64_t SeetaFaceDatabase_Query(struct SeetaFaceDatabase *obj, const SeetaImageData image, const SeetaPointF *points, float *similarity);    // return max index
SEETA_C_API int64_t SeetaFaceDatabase_QueryByCroppedFace(struct SeetaFaceDatabase *obj, const SeetaImageData cropped_face_image, float *similarity);    // return max index
SEETA_C_API size_t SeetaFaceDatabase_QueryTop(struct SeetaFaceDatabase *obj, const SeetaImageData image, const SeetaPointF *points, size_t N, int64_t *index, float *similarity);    // return top N faces
SEETA_C_API size_t SeetaFaceDatabase_QueryTopByCroppedFace(struct SeetaFaceDatabase *obj, const SeetaImageData cropped_face_image, size_t N, int64_t *index, float *similarity);    // return top N faces
SEETA_C_API size_t SeetaFaceDatabase_QueryAbove(struct SeetaFaceDatabase *obj, const SeetaImageData image, const SeetaPointF *points, float threshold, size_t N, int64_t *index, float *similarity);
SEETA_C_API size_t SeetaFaceDatabase_QueryAboveByCroppedFace(struct SeetaFaceDatabase *obj, const SeetaImageData cropped_face_image, float threshold, size_t N, int64_t *index, float *similarity);

SEETA_C_API void SeetaFaceDatabase_RegisterParallel(struct SeetaFaceDatabase *obj, const SeetaImageData image, const SeetaPointF *points, int64_t *index);
SEETA_C_API void SeetaFaceDatabase_RegisterByCroppedFaceParallel(struct SeetaFaceDatabase *obj, const SeetaImageData cropped_face_image, int64_t *index);
SEETA_C_API void SeetaFaceDatabase_Join(struct SeetaFaceDatabase *obj);

SEETA_C_API bool SeetaFaceDatabase_SaveToFile(struct SeetaFaceDatabase *obj, const char *path);
SEETA_C_API bool SeetaFaceDatabase_LoadFromFile(struct SeetaFaceDatabase *obj, const char *path);

SEETA_C_API bool SeetaFaceDatabase_Save(struct SeetaFaceDatabase *obj, SeetaStreamWrite *writer, void *writer_obj);
SEETA_C_API bool SeetaFaceDatabase_Load(struct SeetaFaceDatabase *obj, SeetaStreamRead *reader, void *reader_obj);

#ifdef __cplusplus
}
#endif