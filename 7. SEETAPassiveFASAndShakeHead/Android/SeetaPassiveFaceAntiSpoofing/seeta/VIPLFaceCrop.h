#ifndef _VIPL_FACE_CROP_H
#define _VIPL_FACE_CROP_H

#include "VIPLStruct.h"

namespace VIPL
{

	/**
	 * \brief 人脸裁剪方法
	 */
	enum CROP_METHOD
	{
		BY_LINEAR,	///< 线性插值
		BY_BICUBIC	///< 二次双线性插值
	};

	/**
	 * \brief 获取拥有对应特征点数目的平均人脸模型
	 * \param num 平均人脸模型特征点的数目
	 * \param mean_shape 平均人脸模型的特征点
	 * \param mean_shape_size 平均人脸模型的大小
	 * \param id 平均人脸模型的组别，
	 * \return 对应参数是否有预设平均人脸模型
	 * \note [num, id] 对应获取的平均人脸模型
	 *       [5, 0]: 人脸识别
	 *       [5, 1]: 属性识别
	 */
	VIPL_API bool FaceMeanShape(VIPLPoint *mean_shape, int num, int *mean_shape_size, int id = 0);
	
	/**
	 * \brief 获得缩放后的平均人脸模型
	 * \param num 特征点数量
	 * \param mean_shape 平均人脸模型中的特征点
	 * \param scaler 缩放尺度
	 */
	VIPL_API void ResizeMeanShape(VIPLPoint *mean_shape, int num, double scaler);

	/**
	 * \brief 进行人脸裁剪
	 * \param src_img 原始图像
	 * \param dst_img Crop 好的人脸图像
	 * \param points 人脸的特征点
	 * \param num 人脸特征点个数
	 * \param mean_shape 平均人脸模型
	 * \param mean_shape_size 平均人脸模型的大小
	 * \param method 采样方法
	 * \param final_points 经过变换后的特征点
	 * \param final_size 最后的大小，相对于平均人脸模型
	 * \return 如果 Crop 成功则返回真
	 */
	VIPL_API bool FaceCrop(
		const VIPLImageData &src_img, VIPLImageData &dst_img,
		const VIPLPoint *points, int num,
		const VIPLPoint *mean_shape, int mean_shape_size,
		CROP_METHOD method = BY_LINEAR,
		VIPLPoint *final_points = nullptr,
		int final_size = -1);
}

#endif
