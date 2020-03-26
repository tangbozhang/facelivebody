#pragma once

#include "CStruct.h"

#ifdef __cplusplus

extern "C" {
#endif

struct SeetaPoseEstimator;

enum SeetaPoseEstimator_Axis
{
    SEETA_YAW   = 0,
    SEETA_PITCH = 1,
    SEETA_ROLL  = 2,
};

/**
 * \brief initialize `PoseEstimator`
 * \param setting one specifc model
 * \return new object of `PoseEstimator`, return nullptr if failed
 */
SEETA_C_API struct SeetaPoseEstimator *SeetaNewPoseEstimator(struct SeetaModelSetting setting);

/**
 * \brief free object
 * \param obj object ready to be deleted
 */
SEETA_C_API void SeetaDeletePoseEstimator(const struct SeetaPoseEstimator *obj);

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
SEETA_C_API int SeetaPoseEstimator_SetLogLevel(int level);
SEETA_C_API void SeetaPoseEstimator_SetSingleCalculationThreads(int num);

/**
 * \brief Feed image to `PoseEstimator`
 * \param obj return value by `SeetaNewPoseEstimator`
 * \param image The orginal image
 * \param face The face location
 */
SEETA_C_API void SeetaPoseEstimator_Feed(struct SeetaPoseEstimator *obj, const SeetaImageData image, const SeetaRect face);

/**
 * \brief get angle on given axis
 * \param obj return value by `SeetaNewPoseEstimator`
 * \param axis \sa `Axis`: YAW, PITCH, or ROLL
 * \return angle on given axis
 * \note Must `Feed` image and face first.
 */
SEETA_C_API float SeetaPoseEstimator_Get(struct SeetaPoseEstimator *obj, SeetaPoseEstimator_Axis axis);

/**
 * \brief Get angle from given face on image
 * \param obj return value by `SeetaNewPoseEstimator`
 * \param image The orginal image
 * \param face The face location
 * \param [out] yaw angle on axis yaw
 * \param [out] pitch angle on axis pitch
 * \param [out] roll angle on axis roll
 * \note yaw, pitch or roll can be nullptr
 */
SEETA_C_API void SeetaPoseEstimator_Estimate(struct SeetaPoseEstimator *obj, const SeetaImageData &image, const SeetaRect &face, float *yaw, float *pitch, float *roll);

#ifdef __cplusplus
}
#endif