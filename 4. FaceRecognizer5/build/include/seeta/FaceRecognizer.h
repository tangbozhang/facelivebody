#pragma once

#include "Struct.h"
#include <string>
#include <vector>

#define SEETA_FACE_RECOGNIZER_MAJOR_VERSION 5
#define SEETA_FACE_RECOGNIZER_MINOR_VERSION 0
#define SEETA_FACE_RECOGNIZER_SINOR_VERSION 0

namespace seeta
{
	namespace v500
	{
		class FaceRecognizer
		{
        public:
            class SharedModel;

            SEETA_API explicit FaceRecognizer(const SeetaModelSetting &setting);
            SEETA_API explicit FaceRecognizer(const SharedModel *model);

			SEETA_API ~FaceRecognizer();

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

            SEETA_API static const SharedModel *LoadModel(const SeetaModelSetting &setting);
            SEETA_API static void FreeModel(const SharedModel *model);

            SEETA_API int GetCropFaceWidth() const;
            SEETA_API int GetCropFaceHeight() const;
            SEETA_API int GetCropFaceChannels() const;

            SEETA_API int GetExtractFeatureSize() const;

            SEETA_API bool CropFace(const SeetaImageData &image, const SeetaPointF *points, SeetaImageData &face) const;

            SEETA_API bool ExtractCroppedFace(const SeetaImageData &image, float *features) const;

            SEETA_API bool Extract(const SeetaImageData &image, const SeetaPointF *points, float *features) const;

            SEETA_API float CalculateSimilarity(const float *features1, const float *features2) const;

		private:
			FaceRecognizer(const FaceRecognizer &other) = delete;
			const FaceRecognizer &operator=(const FaceRecognizer &other) = delete;

		private:
            class Implement;
            Implement *m_impl;
		};
	}
	using namespace v500;
}

