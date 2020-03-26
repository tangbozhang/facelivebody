#pragma once

#include "Struct.h"
#include <string>
#include <vector>

namespace seeta
{
	namespace v500
	{
		class FaceComparer
		{
        public:

            SEETA_API explicit FaceComparer(const SeetaModelSetting &setting);

            SEETA_API ~FaceComparer();

            SEETA_API int GetCropFaceWidth() const;
            SEETA_API int GetCropFaceHeight() const;
            SEETA_API int GetCropFaceChannels() const;

            SEETA_API bool CropFace(const SeetaImageData &image, const SeetaPointF *points, SeetaImageData &face) const;

            SEETA_API float CalculateSimilarity(const float *features1, const float *features2) const;

		private:
            FaceComparer(const FaceComparer &other) = delete;
            const FaceComparer &operator=(const FaceComparer &other) = delete;

		private:
            class Implement;
            Implement *m_impl;
		};
	}
	using namespace v500;
}

