#pragma once

#define VIPL_POSE_ESTIMATION_MAJOR_VERSION     1
#define VIPL_POSE_ESTIMATION_MINOR_VERSION     1
#define VIPL_POSE_ESTIMATION_SUBMINOR_VERSION  0

#define _SDK_MIN_MSC_VER 1800
#define _SDK_MAX_MSC_VER 1900
#if defined(_MSC_VER)
#if _MSC_VER < _SDK_MIN_MSC_VER || _MSC_VER > _SDK_MAX_MSC_VER
#error "Unsupported MSVC. Please use VS2013(v120) or compatible VS2015(v140)."
#endif // _MSC_VER < 1800 || _MSC_VER > 1900
#endif // defined(_MSC_VER)

#include <memory>

#include "VIPLStruct.h"

class VIPLPoseEstimationCore;

#if _MSC_VER >= 1600
extern template std::shared_ptr<VIPLPoseEstimationCore>;
#endif

/**
 * \brief 姿态估计类
 */
class VIPLPoseEstimation
{
public:
	/**
	* \brief 模型运行设备
	*/
	enum Device
	{
		AUTO,	/**< 自动检测，会优先使用 GPU */
		CPU,	/**< 使用 CPU 计算 */
		GPU,	/**< 使用 GPU 计算，等价于GPU0 */
		GPU0,   /**< 预定义GPU编号，0号卡 */
		GPU1,
		GPU2,
		GPU3,
		GPU4,
		GPU5,
		GPU6,
		GPU7,
	};

	/**
	 * \brief 姿态估计构造函数，需要在构造的时候传入姿态估计模型
	 * \param [in] model_path 姿态估计模型文件，默认名称为：VIPLPoseEstimation1.0.0.ext.dat
	 * \note 默认会以 AUTO 模式选择计算设备
	 */
	VIPL_API VIPLPoseEstimation(const char *model_path);

	/**
	 * \brief 姿态估计构造函数，需要在构造的时候传入姿态估计模型
	 * \param [in] model_path 姿态估计模型文件，默认名称为：VIPLPoseEstimation1.0.0.ext.dat
	 * \note 默认会以 AUTO 模式选择计算设备
	 */
    VIPL_API VIPLPoseEstimation(const char *model_path, Device device);

    /**
	 * \brief 姿态估计构造函数，可以设定模型版本
	 * \param [in] device 姿态估计模型文件，默认名称为：VIPLPoseEstimation1.0.0.ext.dat
	 * \note 默认会以 AUTO 模式选择计算设备
	 */
    VIPL_API VIPLPoseEstimation(Device device = AUTO);
	
    VIPL_API ~VIPLPoseEstimation();

	/**
	 * \brief 执行姿态估计
	 * \param [in] src_img 输入图像，需要是彩色图像，三个通道以 BGR 次序压缩
	 * \param [in] info 人脸检测信息，为人脸检测的返回值
	 * \param [out] yaw yaw 方向的角度 
	 * \param [out] pitch pitch 方向的角度 
	 * \param [out] roll roll 方向的角度 
	 * \return 只有当姿态估计成功后，才返回真
     * \note 角度的范围为[-90, 90]，当人面向图像采集设备是，各个方向角度为0
	 */
	VIPL_API bool Estimate(const VIPLImageData &src_img, const VIPLFaceInfo &info, float &yaw, float &pitch, float &roll);

private:
	VIPLPoseEstimation(const VIPLPoseEstimation &other) = delete;
	const VIPLPoseEstimation &operator=(const VIPLPoseEstimation &other) = delete;

private:
	std::shared_ptr<VIPLPoseEstimationCore> impl;
};

