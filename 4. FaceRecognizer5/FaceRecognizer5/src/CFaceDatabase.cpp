#include "seeta/CFaceDatabase.h"
#include "seeta/FaceDatabase.h"

struct SeetaFaceDatabase
{
    seeta::FaceDatabase *impl;
};

SeetaFaceDatabase* SeetaNewFaceDatabase(SeetaModelSetting setting)
{
	std::unique_ptr<SeetaFaceDatabase> obj(new SeetaFaceDatabase);
    obj->impl = new seeta::FaceDatabase(setting);
	return obj.release();
}

SeetaFaceDatabase* SeetaNewFaceDatabase2(SeetaModelSetting setting, int extraction_core_number,
    int comparation_core_number)
{
    std::unique_ptr<SeetaFaceDatabase> obj(new SeetaFaceDatabase);
    obj->impl = new seeta::FaceDatabase(setting, extraction_core_number, comparation_core_number);
    return obj.release();
}

void SeetaDeleteFaceDatabase(const SeetaFaceDatabase* obj)
{
	if (!obj) return;
	delete obj->impl;
	delete obj;
}

int SeetaFaceDatabase_SetLogLevel(int level)
{
    return seeta::FaceDatabase::SetLogLevel(level);
}

void SeetaFaceDetector_SetMinFaceSize(SeetaFaceDatabase* obj, int size)
{
}

void SeetaFaceDatabase_SetSingleCalculationThreads(int num)
{
    seeta::FaceDatabase::SetSingleCalculationThreads(num);
}

float SeetaFaceDatabase_Compare(struct SeetaFaceDatabase *obj,
    const SeetaImageData image1, const SeetaPointF* points1,
    const SeetaImageData image2,  const SeetaPointF* points2)
{
    if (!obj || !obj->impl) return 0;
    return obj->impl->Compare(image1, points1, image2, points2);
}

int64_t SeetaFaceDatabase_Register(SeetaFaceDatabase* obj, const SeetaImageData image, const SeetaPointF* points)
{
    if (!obj || !obj->impl) return -1;
    return obj->impl->Register(image, points);
}

int SeetaFaceDatabase_Delete(SeetaFaceDatabase* obj, int64_t index)
{
    if (!obj || !obj->impl) return 0;
    return obj->impl->Delete(index);
}

void SeetaFaceDatabase_Clear(SeetaFaceDatabase* obj)
{
    if (!obj || !obj->impl) return;
    obj->impl->Clear();
}

size_t SeetaFaceDatabase_Count(SeetaFaceDatabase* obj)
{
    if (!obj || !obj->impl) return 0;
    return obj->impl->Count();
}

int64_t SeetaFaceDatabase_Query(SeetaFaceDatabase* obj, const SeetaImageData image, const SeetaPointF* points,
    float* similarity)
{
    if (!obj || !obj->impl) return -1;
    return obj->impl->Query(image, points, similarity);
}

size_t SeetaFaceDatabase_QueryTop(SeetaFaceDatabase* obj, const SeetaImageData image, const SeetaPointF* points,
    size_t N, int64_t* index, float* similarity)
{
    if (!obj || !obj->impl) return 0;
    return obj->impl->QueryTop(image, points, N, index, similarity);
}

void SeetaFaceDatabase_RegisterParallel(SeetaFaceDatabase* obj, const SeetaImageData image, const SeetaPointF* points,
    int64_t* index)
{
    if (!obj || !obj->impl) return;
    obj->impl->RegisterParallel(image, points, index);
}

void SeetaFaceDatabase_Join(SeetaFaceDatabase* obj)
{
    if (!obj || !obj->impl) return;
    obj->impl->Join();
}

bool SeetaFaceDatabase_SaveToFile(SeetaFaceDatabase* obj, const char* path)
{
    if (!obj || !obj->impl) return false;
    return obj->impl->Save(path);
}

bool SeetaFaceDatabase_LoadFromFile(SeetaFaceDatabase* obj, const char* path)
{
    if (!obj || !obj->impl) return false;
    return obj->impl->Load(path);
}

bool SeetaFaceDatabase_Save(SeetaFaceDatabase* obj, SeetaStreamWrite* writer, void* writer_obj)
{
    if (!obj || !obj->impl) return false;
    seeta::CStreamWriter cwriter(writer, writer_obj);
    return obj->impl->Save(cwriter);
}

bool SeetaFaceDatabase_Load(SeetaFaceDatabase* obj, SeetaStreamRead* reader, void* reader_obj)
{
    if (!obj || !obj->impl) return false;
    seeta::CStreamReader creader(reader, reader_obj);
    return obj->impl->Load(creader);
}

size_t SeetaFaceDatabase_QueryAbove(SeetaFaceDatabase* obj, const SeetaImageData image, const SeetaPointF* points,
    float threshold, size_t N, int64_t* index, float* similarity)
{
    if (!obj || !obj->impl) return 0;
    return obj->impl->QueryAbove(image, points, threshold, N, index, similarity);
}

float SeetaFaceDatabase_CompareByCroppedFace(SeetaFaceDatabase* obj, const SeetaImageData cropped_face_image1,
    const SeetaImageData cropped_face_image2)
{
    if (!obj || !obj->impl) return 0;
    return obj->impl->CompareByCroppedFace(cropped_face_image1, cropped_face_image2);
}

int64_t SeetaFaceDatabase_RegisterByCroppedFace(SeetaFaceDatabase* obj, const SeetaImageData cropped_face_image)
{
    if (!obj || !obj->impl) return -1;
    return obj->impl->RegisterByCroppedFace(cropped_face_image);
}

int64_t SeetaFaceDatabase_QueryByCroppedFace(SeetaFaceDatabase* obj, const SeetaImageData cropped_face_image,
    float* similarity)
{
    if (!obj || !obj->impl) return -1;
    return obj->impl->QueryByCroppedFace(cropped_face_image, similarity);
}

size_t SeetaFaceDatabase_QueryTopByCroppedFace(SeetaFaceDatabase* obj, const SeetaImageData cropped_face_image,
    size_t N, int64_t* index, float* similarity)
{
    if (!obj || !obj->impl) return 0;
    return obj->impl->QueryTopByCroppedFace(cropped_face_image, N, index, similarity);
}

size_t SeetaFaceDatabase_QueryAboveByCroppedFace(SeetaFaceDatabase* obj, const SeetaImageData cropped_face_image,
    float threshold, size_t N, int64_t* index, float* similarity)
{
    if (!obj || !obj->impl) return 0;
    return obj->impl->QueryAboveByCroppedFace(cropped_face_image, threshold, N, index, similarity);
}

void SeetaFaceDatabase_RegisterByCroppedFaceParallel(SeetaFaceDatabase* obj, const SeetaImageData cropped_face_image,
    int64_t* index)
{
    if (!obj || !obj->impl) return;
    obj->impl->RegisterByCroppedFaceParallel(cropped_face_image, index);
}

int SeetaFaceDatabase_GetCropFaceWidth()
{
    return seeta::FaceDatabase::GetCropFaceWidth();
}

int SeetaFaceDatabase_GetCropFaceHeight()
{
    return seeta::FaceDatabase::GetCropFaceHeight();
}

int SeetaFaceDatabase_GetCropFaceChannels()
{
    return seeta::FaceDatabase::GetCropFaceChannels();
}

bool SeetaFaceDatabase_CropFace(const SeetaImageData image, const SeetaPointF* points, SeetaImageData* face)
{
    if (!face) return false;
    return seeta::FaceDatabase::CropFace(image, points, *face);
}
