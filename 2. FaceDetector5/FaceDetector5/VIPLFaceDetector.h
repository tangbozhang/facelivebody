#ifndef VIPL_FACE_DETECTOR_H_
#define VIPL_FACE_DETECTOR_H_

#define _SDK_MIN_MSC_VER 1800
#define _SDK_MAX_MSC_VER 1900

#if defined(_MSC_VER)
#if _MSC_VER < _SDK_MIN_MSC_VER || _MSC_VER > _SDK_MAX_MSC_VER
#error "Unsupported MSVC. Please use VS2013(v120) or compatible VS2015(v140)."
#endif // _MSC_VER < 1800 || _MSC_VER > 1900
#endif // defined(_MSC_VER)

#define VIPL_FACE_DETECTOR_MAJOR_VERSION     5
#define VIPL_FACE_DETECTOR_MINOR_VERSION     1
#define VIPL_FACE_DETECTOR_SUBMINOR_VERSION  0

#include <cstdint>
#include <vector>

#include "VIPLStruct.h"

#if _MSC_VER >= 1600
extern template std::vector<VIPLFaceInfo>;
#endif

/** @class VIPLFaceDetector VIPLFaceDetector.h
*  @brief The face detector.
*/
class VIPLFaceDetector
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

	class CoreSize
	{
	public:
		CoreSize() : width(-1), height(-1) {}
		CoreSize(int width, int height) : width(width), height(height) {}
		
		int width;
		int height;
	};

	VIPL_API static void SetNumThreads(int num);
	
	/**
	 * \brief 构造人脸检测器
	 * \param [in] model_path 检测器路径
	 * \note 默认会以 AUTO 模式使用计算设备
	 */
	VIPL_API explicit VIPLFaceDetector(const char* model_path);

	/**
	 * \brief 构造人脸检测器
	 * \param [in] model_path 检测器路径
	 * \param [in] device 使用的计算设备
	 */
	VIPL_API explicit VIPLFaceDetector(const char* model_path, Device device);
	
	/**
	 * \brief 构造人脸检测器
	 * \param [in] model_path 检测器路径
	 * \note 默认会以 AUTO 模式使用计算设备
	 */
	VIPL_API explicit VIPLFaceDetector(const char* model_path, const CoreSize &core_size);

	/**
	 * \brief 构造人脸检测器
	 * \param [in] model_path 检测器路径
	 * \param [in] device 使用的计算设备
	 */
	VIPL_API explicit VIPLFaceDetector(const char* model_path, const CoreSize &core_size, Device device);
    
	VIPL_API ~VIPLFaceDetector();
    
	/**
	 * \brief 检测人脸
	 * \param [in] img 输入图像，需要 RGB 彩色通道
	 * \return 检测到的人脸（VIPLFaceInfo）数组
	 * \note 此函数不支持多线程调用，在多线程环境下需要建立对应的 VIPLFaceDetector 的对象分别调用检测函数
	 * \seet VIPLFaceInfo, VIPLImageData 
	 */
	VIPL_API std::vector<VIPLFaceInfo> Detect(const VIPLImageData & img);

	/**
	 * \brief 设置最小人脸
	 * \param [in] size 最小可检测的人脸大小，为人脸宽和高乘积的二次根值
	 * \note 最下人脸为 20，小于 20 的值会被忽略 
	 */
	VIPL_API void SetMinFaceSize(int32_t size);
    
	/**
	 * \brief 设置图像金字塔的缩放比例
	 * \param [in] factor 缩放比例
	 * \note 该值最小为 1.414，小于 1.414 的值会被忽略
	 */
	VIPL_API void SetImagePyramidScaleFactor(float factor);

	/**
	 * \brief 设置级联网路网络的三级阈值
	 * \param [in] thresh1 第一级阈值
	 * \param [in] thresh2 第二级阈值
	 * \param [in] thresh3 第三级阈值
	 * \note 默认推荐为：0.62, 0.47, 0.985
	 */
    VIPL_API void SetScoreThresh(float thresh1, float thresh2, float thresh3);

    /**
    * \brief 获取最小人脸
    * \return size 最小可检测的人脸大小，为人脸宽和高乘积的二次根值
    */
    VIPL_API int32_t GetMinFaceSize() const;

    /**
    * \brief 获取图像金字塔的缩放比例
    * \return factor 缩放比例
    */
    VIPL_API float GetImagePyramidScaleFactor() const;

    /**
     * \brief 获取级联网路网络的三级阈值
     * \param [out] thresh1 第一级阈值
     * \param [out] thresh2 第二级阈值
     * \param [out] thresh3 第三级阈值
     * \note 可以设置为 nullptr，表示不取该值
     */
    VIPL_API void GetScoreThresh(float *thresh1, float *thresh2, float *thresh3) const;

	/**
	 * \brief 是否以稳定模式输出人脸检测结果
	 * \param stable 是否稳定
	 * \note 默认是不以稳定模型工作的
	 * \note 只有在视频中连续跟踪时，才使用此方法
	 */
	VIPL_API void SetVideoStable(bool stable = true);

	/**
	 * \brief 获取当前是否是稳定工作模式
	 * \return 是否稳定
	 */
	VIPL_API bool GetVideoStable() const;

	VIPL_API CoreSize GetCoreSize() const;

private:
	VIPLFaceDetector(const VIPLFaceDetector &other) = delete;
	const VIPLFaceDetector &operator=(const VIPLFaceDetector &other) = delete;

private:
    void* impl_;
};

#endif  // VIPL_FACE_DETECTOR_H_
