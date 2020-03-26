#pragma once

#include "Struct.h"
#include <string>
#include <vector>

#define SEETA_POSE_ESTIMATOR_MAJOR_VERSION 1
#define SEETA_POSE_ESTIMATOR_MINOR_VERSION 1
#define SEETA_POSE_ESTIMATOR_SINOR_VERSION 0

namespace seeta
{
	namespace v110
	{
		class PoseEstimator
		{
		public:
            enum Axis
            {
                YAW     = 0,
                PITCH   = 1,
                ROLL    = 2,
            };

		    /**
			 * \brief initialize `PoseEstimator`
			 * \param setting one specifc model, or zero model
			 */
			SEETA_API explicit PoseEstimator(const SeetaModelSetting &setting = seeta::ModelSetting());

			SEETA_API ~PoseEstimator();

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

		    /**
             * \brief Feed image to `PoseEstimator`
             * \param image The orginal image
             * \param face The face location
             */
            SEETA_API void Feed(const SeetaImageData &image, const SeetaRect &face) const;

		    /**
             * \brief get angle on given axis
             * \param axis \sa `Axis`: YAW, PITCH, or ROLL
             * \return angle on given axis
             * \note Must `Feed` image and face first. 
             */
            SEETA_API float Get(Axis axis) const;

		    /**
             * \brief Get angle from given face on image
             * \param image The orginal image
             * \param face The face location
             * \param [out] yaw angle on axis yaw
             * \param [out] pitch angle on axis pitch
             * \param [out] roll angle on axis roll
             * \note yaw, pitch or roll can be nullptr
             */
            SEETA_API void Estimate(const SeetaImageData &image, const SeetaRect &face, float *yaw, float *pitch, float *roll) const;

		private:
			PoseEstimator(const PoseEstimator &other) = delete;
			const PoseEstimator &operator=(const PoseEstimator &other) = delete;

		private:
            class Implement;
            Implement *m_impl;
		};
	}
	using namespace v110;
}

