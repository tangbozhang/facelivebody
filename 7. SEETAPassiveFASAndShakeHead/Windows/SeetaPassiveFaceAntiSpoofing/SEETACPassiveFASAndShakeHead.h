#ifndef SEETA_C_PASSIVE_FAS_AND_SHAKE_HEAD_H
#define SEETA_C_PASSIVE_FAS_AND_SHAKE_HEAD_H

#include "VIPLStruct.h"

#define VIPL_C_API extern "C" VIPL_API //支持C++代码调用C语言代码

struct SEETACPassiveFASAndShakeHead;

enum SeetaCSystemState
{
	NO_FACE,
	DETECTING,
	PLEASE_TURN,
	PASSED,
	NOT_PASSED,
	PLEASE_BLINK,
	PLEASE_NOD_HEAD,
    PLEASE_FACE_THE_CAMERA
}; //系统当前的检测状态 （分别为没有人脸，正在检测，请转头，通过，未通过,）

VIPL_C_API SEETACPassiveFASAndShakeHead* SEETACPassiveFASAndShakeHead_New(const char* ModelPath,
                                                                          const int shakeHeadAngleThreshold,
                                                                          const int nodHeadAngleThreshold,
                                                                          const double clarityTheshold,
                                                                          const double fuseThreshold,
                                                                          SeetaCSystemState* systemState,
                                                                          const int firstPhaseFrameNum,
                                                                          const int detectFrameNum);
VIPL_C_API void SEETACPassiveFASAndShakeHead_Delete(SEETACPassiveFASAndShakeHead* object);

/**
 * \brief 活体检测
 * \param object SEETACPassiveFASAndShakeHead_New 的返回值
 * \param VIPLImage 输入图像
 * \param info 人脸位置
 * \param points5 特征点
 * \param systemState 系统状态
 */
VIPL_C_API void SEETACPassiveFASAndShakeHead_detection(SEETACPassiveFASAndShakeHead* object, const VIPLImageData* VIPLImage, const VIPLFaceInfo* info,
                                                      const VIPLPoint* points5, SeetaCSystemState* systemState);

/**
 * \brief 重置系统状态，开始下一步检测
 * \param object SEETACPassiveFASAndShakeHead_New 的返回值
 * \param systemState 
 */
VIPL_C_API void SEETACPassiveFASAndShakeHead_reset(SEETACPassiveFASAndShakeHead* object, SeetaCSystemState* systemState);

/**
 * \brief 设置要检测的动作序列，可输入 pleaseTrun 和 pleaseBlink
 * \param object SEETACPassiveFASAndShakeHead_New 的返回值
 * \param actions 设置动作的数组
 * \param actions_len 设置动作数组的长度
 * \return 设置动作是否成功
 * \note 默认动作只有 pleaseTrun。如果输入了不支持的动作，将返回假
 */
VIPL_C_API int SEETACPassiveFASAndShakeHead_set_actions(SEETACPassiveFASAndShakeHead* object, const SeetaCSystemState* actions, int actions_len);

/**
 * \brief 获取日志用于调试
 * \param object SEETACPassiveFASAndShakeHead_New 的返回值
 * \param value1 
 * \param value2 
 * \param yawAngle 
 */
VIPL_C_API void SEETACPassiveFASAndShakeHead_getLog(SEETACPassiveFASAndShakeHead* object, double* value1, double* value2, double* yawAngle,
                                                    double* pitchAngle);
/**
 * \brief 返回识别过程中姿态最正面的人脸图片
 * \param object SEETACPassiveFASAndShakeHead_New 的返回值
 * \param [out] snapshot 抓拍图片
 */
VIPL_C_API void SEETACPassiveFASAndShakeHead_getSnapshot(SEETACPassiveFASAndShakeHead* object, VIPLImageData *snapshot);
/**
 * \brief 返回识别过程中姿态最正面的人脸图片
 * \param object SEETACPassiveFASAndShakeHead_New 的返回值
 * \param [out] snapshot 抓拍图片
 */
VIPL_C_API void SEETACPassiveFASAndShakeHead_getSnapshot_v2(SEETACPassiveFASAndShakeHead* object, VIPLImageData *snapshot, VIPLFaceInfo *face);

#endif	// SEETA_C_PASSIVE_FAS_AND_SHAKE_HEAD_H
