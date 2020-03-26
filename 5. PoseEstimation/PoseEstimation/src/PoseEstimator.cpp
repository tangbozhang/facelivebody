#include "seeta/PoseEstimator.h"

#include <orz/utils/log.h>
#include "seeta/ForwardNet.h"
#include "VIPLPoseEstimation.h"

#define VER_HEAD(x) #x "."
#define VER_TAIL(x) #x
#define GENERATE_VER(seq) FUN_MAJOR seq
#define FUN_MAJOR(x) VER_HEAD(x) FUN_MINOR
#define FUN_MINOR(x) VER_HEAD(x) FUN_SINOR
#define FUN_SINOR(x) VER_TAIL(x)

#define LIBRARY_VERSION GENERATE_VER( \
	(SEETA_POSE_ESTIMATOR_MAJOR_VERSION) \
	(SEETA_POSE_ESTIMATOR_MINOR_VERSION) \
	(SEETA_POSE_ESTIMATOR_SINOR_VERSION))

#define LIBRARY_NAME "PoseEstimator"

#define LOG_HEAD LIBRARY_NAME "(" LIBRARY_VERSION "): "

#include <orz/io/i.h>
#include <map>
#if SEETA_LOCK_SDK
#include <lock/macro.h>
#endif

namespace seeta
{
    namespace v110
    {
        class PoseEstimator::Implement
        {
        public:
            Implement(const SeetaModelSetting& setting)
            {
                VIPLPoseEstimation::Device device;
                switch (setting.device)
                {
                case SEETA_DEVICE_AUTO:
                    device = VIPLPoseEstimation::AUTO;
                    break;
                case SEETA_DEVICE_CPU:
                    device = VIPLPoseEstimation::CPU;
                    break;
                case SEETA_DEVICE_GPU:
                    device = VIPLPoseEstimation::Device(VIPLPoseEstimation::GPU0 + setting.id);
                    break;
                default:
                    device = VIPLPoseEstimation::AUTO;
                    break;
                }
                seeta::ModelSetting exciting = setting;
                auto models = exciting.get_model();
                if (models.size() > 1)
                {
                    orz::Log(orz::FATAL) << LOG_HEAD << "Must input 1 or 0 model." << orz::crash;
                }
                else if (models.size() == 1)
                {
                    orz::Log(orz::STATUS) << LOG_HEAD << "Loading models...";
                    m_impl.reset(new VIPLPoseEstimation(models[0].c_str(), device));
                }
                else
                {
                    m_impl.reset(new VIPLPoseEstimation(device));
                }
            }

            void Feed(const SeetaImageData &image, const SeetaRect &face)
            {
                VIPLImageData vimage(image.width, image.height, image.channels);
                vimage.data = image.data;
                VIPLFaceInfo vface;
                vface.x = face.x;
                vface.y = face.y;
                vface.width = face.width;
                vface.height = face.height;
                m_impl->Estimate(vimage, vface, m_angle_yaw, m_angle_pitch, m_angle_roll);
            }

            std::shared_ptr<VIPLPoseEstimation> m_impl;

            float m_angle_yaw = 0;
            float m_angle_pitch = 0;
            float m_angle_roll = 0;
        };

        PoseEstimator::PoseEstimator(const SeetaModelSetting& setting)
            : m_impl(new Implement(setting))
        {
        }

        PoseEstimator::~PoseEstimator()
        {
            delete m_impl;
        }

        int PoseEstimator::SetLogLevel(int level)
        {
            return orz::GlobalLogLevel(orz::LogLevel(level));
        }

        void PoseEstimator::SetSingleCalculationThreads(int num)
        {
            return ForwardNet::SetSingleCalculationThreads(num);
        }

        void PoseEstimator::Feed(const SeetaImageData& image, const SeetaRect& face) const
        {
            m_impl->Feed(image, face);
        }

        float PoseEstimator::Get(Axis axis) const
        {
            switch (axis)
            {
            case YAW:   return m_impl->m_angle_yaw;
            case PITCH: return m_impl->m_angle_pitch;
            case ROLL:  return m_impl->m_angle_roll;
            default: return 0;
            }
        }

        void PoseEstimator::Estimate(const SeetaImageData& image, const SeetaRect& face, float* yaw, float* pitch,
            float* roll) const
        {
            m_impl->Feed(image, face);
            if (yaw)    *yaw    = m_impl->m_angle_yaw;
            if (pitch)  *pitch  = m_impl->m_angle_pitch;
            if (roll)   *roll   = m_impl->m_angle_roll;
        }
    }
}

