#pragma once

#include "VIPLStruct.h"
#include <string>
#include <memory>
#include <vector>

enum SystemState{ noFace, detecting, pleaseTrun, passed, notPass, pleaseBlink,pleaseNodHead, pleaseFaceTheCamera};	//系统当前的检测状态 （分别为没有人脸，正在检测，请转头，通过，未通过,）


class FusionFASProcess;
class SEETAPassiveFASAndShakeHead
{
public:
    class ModelSetting
    {
    public:
        std::string FAS;
        std::string PE;
        std::string EBD;
        std::string FR;
    };

    class Buffer
    {
    public:
        Buffer() {}
        Buffer(void *data, size_t size) : data(data), size(size) {}

        void *data = nullptr;
        size_t size = 0;
    };

    class BufferSetting
    {
    public:
        Buffer FAS;
        Buffer PE;
        Buffer EBD;
        Buffer FR;
    };

    VIPL_API SEETAPassiveFASAndShakeHead(const std::string ModelPath,
        const int shakeHeadAngleThreshold,
        const int nodHeadAngleThreshold,
        const double clarityTheshold,
        const double fuseThreshold,
        SystemState &systemState,
        const int firstPhaseFrameNum,
        const int detectFrameNum);

    VIPL_API SEETAPassiveFASAndShakeHead(const ModelSetting &model_setting,
        const int shakeHeadAngleThreshold,
        const int nodHeadAngleThreshold,
        const double clarityTheshold,
        const double fuseThreshold,
        SystemState &systemState,
        const int firstPhaseFrameNum,
        const int detectFrameNum);
	VIPL_API ~SEETAPassiveFASAndShakeHead();

	/**
	 * \brief 活体检测
	 * \param VIPLImage 输入图像
	 * \param info 人脸位置
	 * \param points5 特征点
	 * \param systemState 系统状态
	 */
	VIPL_API void detecion(VIPLImageData VIPLImage, const VIPLFaceInfo &info, const VIPLPoint *points5, SystemState &systemState);

	/**
	 * \brief 重置系统状态，开始下一步检测
	 * \param systemState 
	 */
	VIPL_API void reset(SystemState &systemState);

	/**
	 * \brief 设置要检测的动作序列，可输入 pleaseTrun 和 pleaseBlink
	 * \param actions 
	 * \return 设置动作是否成功
	 * \note 默认动作只有 pleaseTrun。如果输入了不支持的动作，将返回假
	 */
	VIPL_API bool set_actions(const std::vector<SystemState>& actions);

	/**
	 * \brief 获取日志用于调试
	 * \param value1 
	 * \param value2 
	 * \param yawAngle 
	 */
	VIPL_API void getLog(double &value1, double &value2, double &yawAngle, double &pitchAngle);
	/**
	 * \brief 返回识别过程中姿态最正面的人脸图片
	 * \param 
	 */
    VIPL_API VIPLImageData getSnapshot();

    /**
     * \brief 返回识别过程中姿态最正面的人脸图片
     * \param
     */
    VIPL_API VIPLImageData getSnapshot(VIPLFaceInfo &face);
private:
	SEETAPassiveFASAndShakeHead(const SEETAPassiveFASAndShakeHead &other) = delete;
	const SEETAPassiveFASAndShakeHead &operator=(const SEETAPassiveFASAndShakeHead &other) = delete;
	std::shared_ptr<FusionFASProcess> impl;
};

