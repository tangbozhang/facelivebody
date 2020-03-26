#pragma once

#include "Struct.h"
#include "CFaceInfo.h"

#include <string>
#include <vector>

#define SEETA_FACE_DETECTOR_MAJOR_VERSION 5
#define SEETA_FACE_DETECTOR_MINOR_VERSION 1
#define SEETA_FACE_DETECTOR_SINOR_VERSION 0

namespace seeta
{
	namespace v510
	{

		class FaceDetector
		{
		public:
			/**
			 * \brief 加载模型文件
			 * \param setting 模型文件
			 */
			SEETA_API explicit FaceDetector(const SeetaModelSetting &setting);
			/**
			 * \brief 加载模型文件，并设置计算内核大小
			 * \param setting 模型文件
			 * \param core_size 设置计算内核大小，越小占用资源越少
			 */
			SEETA_API explicit FaceDetector(const SeetaModelSetting &setting, const SeetaSize &core_size);
            SEETA_API ~FaceDetector();

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
			 * \brief 检测人脸
			 * \param [in] image 输入图像，需要 RGB 彩色通道
			 * \return 检测到的人脸（SeetaRotateFaceInfo）数组
			 * \note 此函数不支持多线程调用，在多线程环境下需要建立对应的 FaceDetector 的对象分别调用检测函数
			 * \seet SeetaRotateFaceInfo, SeetaImageData
			 */
            SEETA_API SeetaFaceInfoArray Detect(const SeetaImageData &image) const;

			/**
			 * \brief 设置最小人脸
			 * \param [in] size 最小可检测的人脸大小，为人脸宽和高乘积的二次根值
			 * \note 最下人脸为 20，小于 20 的值会被忽略
			 */
			SEETA_API void SetMinFaceSize(int32_t size);

			/**
			 * \brief 获取最小人脸
			 * \return 最小可检测的人脸大小，为人脸宽和高乘积的二次根值
			 */
			SEETA_API int32_t GetMinFaceSize() const;

			/**
			 * \brief 设置图像金字塔的缩放比例
			 * \param [in] factor 缩放比例
			 * \note 该值最小为 1.414，小于 1.414 的值会被忽略
			 */
			SEETA_API void SetImagePyramidScaleFactor(float factor);

			/**
			 * \brief 获取图像金字塔的缩放比例
			 * \return 缩放比例
			 */
            SEETA_API float GetImagePyramidScaleFactor() const;

			/**
			 * \brief 设置级联网路网络的三级阈值
			 * \param [in] thresh1 第一级阈值
			 * \param [in] thresh2 第二级阈值
			 * \param [in] thresh3 第三级阈值
			 * \note 默认推荐为：0.67, 0.4, 0.95
			 */
			SEETA_API void SetScoreThresh(float thresh1, float thresh2, float thresh3);

			/**
			 * \brief 获取级联网路网络的三级阈值
			 * \param [out] thresh1 第一级阈值
			 * \param [out] thresh2 第二级阈值
			 * \param [out] thresh3 第三级阈值
			 * \note 可以设置为 nullptr，表示不取该值
			 */
			SEETA_API void GetScoreThresh(float *thresh1, float *thresh2, float *thresh3) const;

			/**
			 * \brief 是否以稳定模式输出人脸检测结果
			 * \param stable 是否稳定
			 * \note 默认是不以稳定模型工作的
			 * \note 只有在视频中连续跟踪时，才使用此方法
			*/
			SEETA_API void SetVideoStable(bool stable = true);

			/**
			 * \brief 获取当前是否是稳定工作模式
			 * \return 是否稳定
			 */
			SEETA_API bool GetVideoStable() const;

		private:
			FaceDetector(const FaceDetector &other) = delete;
			const FaceDetector &operator=(const FaceDetector &other) = delete;

		private:
			void *m_impl;
		};
	}
    using namespace v510;
}
