#pragma once

#include "Struct.h"
#include <string>
#include <vector>

#define SEETA_POINT_DETECTOR_MAJOR_VERSION 5
#define SEETA_POINT_DETECTOR_MINOR_VERSION 0
#define SEETA_POINT_DETECTOR_SINOR_VERSION 0

namespace seeta
{
	namespace v500
	{
		class PointDetector
		{
		public:
			SEETA_API explicit PointDetector(const SeetaModelSetting &setting);
            SEETA_API ~PointDetector();

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

			SEETA_API int GetLandmarkNumber() const;

			/**
			 * \brief detect points
			 * \param image source image
			 * \param face face location
			 * \param points array for output points, with size `GetLandmarkNumber()`
			 * \return return false if failed
			 */
            SEETA_API bool Detect(const SeetaImageData &image, const SeetaRect &face, SeetaPointF *points) const;

            /**
             * \brief detect points
             * \param image source image
             * \param face face location
             * \param points array for output points, with size `GetLandmarkNumber()`
             * \param masks array for point's mask, with size `GetLandmarkNumber()`. Non zero for the point has been covered, 0 for not.
             * \return return false if failed
             */
            SEETA_API bool Detect(const SeetaImageData &image, const SeetaRect &face, SeetaPointF *points, int *masks) const;

		private:
			PointDetector(const PointDetector &other) = delete;
			const PointDetector &operator=(const PointDetector &other) = delete;

		private:
			void *m_impl;
		};
	}
	using namespace v500;
}

