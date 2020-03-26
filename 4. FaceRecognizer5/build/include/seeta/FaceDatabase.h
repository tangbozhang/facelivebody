#pragma once

#include "Struct.h"
#include "Stream.h"

#include <string>
#include <vector>

#define SEETA_FACE_DATABASE_MAJOR_VERSION 5
#define SEETA_FACE_DATABASE_MINOR_VERSION 0
#define SEETA_FACE_DATABASE_SINOR_VERSION 0

namespace seeta
{
	namespace v500
    {
        class FaceRecognizer;
        /**
         * \brief Only support single thread running
         */
		class FaceDatabase
		{
		public:
			SEETA_API explicit FaceDatabase(const SeetaModelSetting &setting);
			SEETA_API explicit FaceDatabase(const SeetaModelSetting &setting, int extraction_core_number, int comparation_core_number);
			SEETA_API ~FaceDatabase();

			/**
			 * \brief 
			 * \param level 
			 * \return older level setting
			 * \note level:
			 *  DEBUG = 1,
			 *  STATUS = 2,
			 *  INFO = 3,
			 *  FATAL = 4,
			 */
            SEETA_API static int SetLogLevel(int level);

            SEETA_API static void SetSingleCalculationThreads(int num);

            SEETA_API static int GetCropFaceWidth();
            SEETA_API static int GetCropFaceHeight();
            SEETA_API static int GetCropFaceChannels();

            SEETA_API static bool CropFace(const SeetaImageData &image, const SeetaPointF *points, SeetaImageData &face);

            SEETA_API float Compare(
                const SeetaImageData &image1, const SeetaPointF *points1,
                const SeetaImageData &image2, const SeetaPointF *points2) const;

            SEETA_API float CompareByCroppedFace(
                const SeetaImageData &cropped_face_image1,
                const SeetaImageData &cropped_face_image2) const;

            SEETA_API int64_t Register(const SeetaImageData &image, const SeetaPointF *points);
            SEETA_API int64_t RegisterByCroppedFace(const SeetaImageData &cropped_face_image);
            SEETA_API int Delete(int64_t index);    // return effected lines, 1 for succeed, 0 for nothing
            SEETA_API void Clear(); // clear all faces

            SEETA_API size_t Count() const;
            SEETA_API int64_t Query(const SeetaImageData &image, const SeetaPointF *points, float *similarity = nullptr) const;    // return max index
            SEETA_API int64_t QueryByCroppedFace(const SeetaImageData &cropped_face_image, float *similarity = nullptr) const;    // return max index
            /**
             * \biref query top N face
             * \param image The original image
             * \param points Pointer to 5 SeetaPoints
             * \param N Querying top N
             * \param [out] index Pointer to array saving result, have at least N space.
             * \param [out] similarity Pointer to array saving result, have at least N space.
             * \return Queried face number
             */
            SEETA_API size_t QueryTop(const SeetaImageData &image, const SeetaPointF *points, size_t N, int64_t *index, float *similarity) const;    // return top N faces
            SEETA_API size_t QueryTopByCroppedFace(const SeetaImageData &cropped_face_image, size_t N, int64_t *index, float *similarity) const;    // return top N faces

            /**
            * \biref query all faces similarity greater than threhold
            * \param image The original image
            * \param points Pointer to 5 SeetaPoints
            * \param threshold Threshold
            * \param N size of `index` and `similarity`
            * \param [out] index Pointer to array saving result, have at least N space.
            * \param [out] similarity Pointer to array saving result, have at least N space.
            * \return Queried face number
            */
            SEETA_API size_t QueryAbove(const SeetaImageData &image, const SeetaPointF *points, float threshold, size_t N, int64_t *index, float *similarity) const;
            SEETA_API size_t QueryAboveByCroppedFace(const SeetaImageData &cropped_face_image, float threshold, size_t N, int64_t *index, float *similarity) const;

            SEETA_API void RegisterParallel(const SeetaImageData &image, const SeetaPointF *points, int64_t *index);
            SEETA_API void RegisterByCroppedFaceParallel(const SeetaImageData &cropped_face_image, int64_t *index);
            SEETA_API void Join() const;

            SEETA_API bool Save(const char *path) const;
            SEETA_API bool Load(const char *path);

            SEETA_API bool Save(StreamWriter &writer) const;
            SEETA_API bool Load(StreamReader &reader);

            SEETA_API FaceRecognizer *ExtractionCore(int i = 0);

		private:
			FaceDatabase(const FaceDatabase &other) = delete;
			const FaceDatabase &operator=(const FaceDatabase &other) = delete;

		private:
            class Implement;
            Implement *m_impl;
		};
	}
	using namespace v500;
}

