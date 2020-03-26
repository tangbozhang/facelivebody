#ifndef _SEETA_FACE_COMPARER_H
#define _SEETA_FACE_COMPARER_H

#include "VIPLStruct.h"
#include <vector>

class VIPLFaceComparer {
public:
    VIPL_API explicit VIPLFaceComparer(const char* modelPath = NULL);

    VIPL_API ~VIPLFaceComparer();


    /**
    * \brief 为识别器加载模型，会卸载原加载模型
    * \param [in] modelPath 识别模型路径
    * \return 加载成功后返回真
    * \note 此函数是为了在构造的时候没有加载模型的情况下调用
    * \note 默认会以 AUTO 模式使用计算设备
    */
    VIPL_API bool LoadModel(const char* modelPath);

	/**
	 * \brief 计算特征 fc1 和 fc2 的相似度
	 * \param [in] fc1 特征向量1
	 * \param [in] fc2 特征向量2
	 * \param [in] dim 特征维度
	 * \return 相似度
	 * \note 默认特征维度应该是 GetFeatureSize() 的返回值，如果不是，则需要传入对应的特征长度
	 */
	VIPL_API float CalcSimilarity(const float *fc1, const float *fc2, long dim = -1);

	/**
	 * \brief 计算特征 fc1 和 fc2 的相似度
	 * \param [in] fc1 特征向量1（归一化的）
	 * \param [in] fc2 特征向量2（归一化的）
	 * \param [in] dim 特征维度
	 * \return 相似度
	 * \note 默认特征维度应该是 GetFeatureSize() 的返回值，如果不是，则需要传入对应的特征长度
	 * \note 此计算相似度函数必须输入归一化的特征向量，由后缀 Normalized 的函数提取的特征
	 */
    VIPL_API float CalcSimilarityNormalized(const float *fc1, const float *fc2, long dim = -1);

    /**
    * \brief 获取用于识别而裁剪的人脸图片宽度
    * \return 人脸图片宽度
    */
    VIPL_API uint32_t GetCropWidth();

    /**
    * \brief 获取用于识别而裁剪的人脸图片高度
    * \return 人脸图片高度
    */
    VIPL_API uint32_t GetCropHeight();

    /**
    * \brief 获取用于识别而裁剪的人脸图片通道数
    * \return 人脸图片通道数
    */
    VIPL_API uint32_t GetCropChannels();

    /**
    * \brief 裁剪人脸
    * \param [in] srcImg 原始图像，彩色
    * \param [in] llpoint 原始图像中人脸特征点，5个
    * \param [out] dstImg 目标图像，根据裁剪信息预先申请好内存
    * \param [in] posNum 人脸姿态，保留参数，此版本无意义
    * \return 只有裁剪成功后才返回真
    */
    VIPL_API bool CropFace(const VIPLImageData &srcImg,
        const VIPLPoint (&llpoint)[5],
        const VIPLImageData &dstImg,
        uint8_t posNum = 1);

private:
    VIPLFaceComparer(const VIPLFaceComparer &other) = delete;
    const VIPLFaceComparer &operator=(const VIPLFaceComparer &other) = delete;

private:
	class Data;
    Data* data;
};
#endif // _SEETA_FACE_COMPARER_H
