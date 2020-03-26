#pragma once

#define VIPL_EYE_BLINK_DETECOTR_MAJOR_VERSION     1
#define VIPL_EYE_BLINK_DETECOTR_MINOR_VERSION     3
#define VIPL_EYE_BLINK_DETECOTR_SUBMINOR_VERSION  0

#define _SDK_MIN_MSC_VER 1800
#define _SDK_MAX_MSC_VER 1900
#if defined(_MSC_VER)
#if _MSC_VER < _SDK_MIN_MSC_VER || _MSC_VER > _SDK_MAX_MSC_VER
#error "Unsupported MSVC. Please use VS2013(v120) or compatible VS2015(v140)."
#endif // _MSC_VER < 1800 || _MSC_VER > 1900
#endif // defined(_MSC_VER)

#include "VIPLStruct.h"
#include <memory>
#include <string>
#include <array>

class VIPLEyeBlinkDetectorCore;

#if _MSC_VER >= 1600
extern template std::shared_ptr<VIPLEyeBlinkDetectorCore>;
extern template std::array<VIPLPoint, 5>;
#endif

class VIPLEyeBlinkDetector
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

	static const int LEFT_EYE = 1;
	static const int RIGHT_EYE = 2;

	VIPL_API VIPLEyeBlinkDetector(const char *model_path, Device device = AUTO);
	VIPL_API ~VIPLEyeBlinkDetector();

	/**
	 * \brief 检测眨眼
	 * \param image 原始彩色图像
	 * \param points 定位的特征点，需要前两个点分别表示左眼和右眼坐标
	 * \return 返回二进制数表示是否眨眼，左眼闭上会把LEFT_EYE位置为1，右眼闭上会把RIGHT_EYE位置为1，
	 */
	VIPL_API int Detect(const VIPLImageData &image, const VIPLPoint *points);

	/**
	 * 检测当前状态是否闭了双眼，用于一般的眨眼检测.
	 * \return true为闭眼
	 */
	VIPL_API bool ClosedEyes(const VIPLImageData &image, const VIPLPoint *points);

private:
	VIPLEyeBlinkDetector(const VIPLEyeBlinkDetector &other) = delete;
	const VIPLEyeBlinkDetector &operator=(const VIPLEyeBlinkDetector &other) = delete;

private:
	std::shared_ptr<VIPLEyeBlinkDetectorCore> impl;
};

