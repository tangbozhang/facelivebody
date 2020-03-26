#include "seeta/FaceDetector.h"
#include "VIPLFaceDetector.h"
#include <orz/utils/log.h>
#include "seeta/ForwardNet.h"

#define VER_HEAD(x) #x "."
#define VER_TAIL(x) #x
#define GENERATE_VER(seq) FUN_MAJOR seq
#define FUN_MAJOR(x) VER_HEAD(x) FUN_MINOR
#define FUN_MINOR(x) VER_HEAD(x) FUN_SINOR
#define FUN_SINOR(x) VER_TAIL(x)

#define LIBRARY_VERSION GENERATE_VER( \
	(SEETA_FACE_DETECTOR_MAJOR_VERSION) \
	(SEETA_FACE_DETECTOR_MINOR_VERSION) \
	(SEETA_FACE_DETECTOR_SINOR_VERSION))

#define LIBRARY_NAME "FaceDetector"

#define LOG_HEAD LIBRARY_NAME "(" LIBRARY_VERSION "): "

namespace seeta
{
    namespace v510
    {
        class InnerFaceDetectorV5
        {
        public:
            InnerFaceDetectorV5(const SeetaModelSetting &setting, const SeetaSize &core_size = Size(-1, -1))
            {
                std::vector<std::string> model;
                int index = 0;
                while (setting.model[index]) model.push_back(setting.model[index++]);
                if (model.empty())
                {
                    orz::Log(orz::FATAL) << LOG_HEAD << "Must input 1 model." << orz::crash;
                }
                VIPLFaceDetector::Device device;
                switch (setting.device)
                {
                case SEETA_DEVICE_AUTO:
                    device = VIPLFaceDetector::AUTO;
                    break;
                case SEETA_DEVICE_CPU:
                    device = VIPLFaceDetector::CPU;
                    break;
                case SEETA_DEVICE_GPU:
                    device = VIPLFaceDetector::Device(VIPLFaceDetector::GPU0 + setting.id);
                    break;
                default:
                    device = VIPLFaceDetector::AUTO;
                    break;
                }
                m_impl.reset(new VIPLFaceDetector(model[0].c_str(), VIPLFaceDetector::CoreSize(core_size.width, core_size.height), device));
            }

            SeetaFaceInfoArray Detect(const SeetaImageData &image) const
            {
                VIPLImageData vimage(image.width, image.height, image.channels);
                vimage.data = image.data;
                auto faces = m_impl->Detect(vimage);
                // sort
                std::sort(faces.begin(), faces.end(), [](const VIPLFaceInfo &lhs, const VIPLFaceInfo &rhs)
                {
                    return lhs.width * lhs.height > rhs.width * rhs.height;
                });
                m_faces.resize(faces.size());
                for (size_t i = 0; i < faces.size(); ++i)
                {
                    m_faces[i].pos.x = faces[i].x;
                    m_faces[i].pos.y = faces[i].y;
                    m_faces[i].pos.width = faces[i].width;
                    m_faces[i].pos.height = faces[i].height;
                    m_faces[i].score = float(faces[i].score);
                }
                struct SeetaFaceInfoArray seeta_faces;
                seeta_faces.data = m_faces.data();
                seeta_faces.size = int(m_faces.size());
                return seeta_faces;
            }

            std::shared_ptr<VIPLFaceDetector> m_impl;
            mutable std::vector<SeetaFaceInfo> m_faces;
        };

        FaceDetector::FaceDetector(const SeetaModelSetting& setting)
            : m_impl(new InnerFaceDetectorV5(setting))
        {

        }

	    FaceDetector::FaceDetector(const SeetaModelSetting& setting, const SeetaSize& core_size)
			: m_impl(new InnerFaceDetectorV5(setting, core_size))
	    {
	    }

	    FaceDetector::~FaceDetector()
        {
            auto impl = reinterpret_cast<InnerFaceDetectorV5*>(this->m_impl);
            delete impl;
        }

        int FaceDetector::SetLogLevel(int level)
        {
            return orz::GlobalLogLevel(orz::LogLevel(level));
        }

        void FaceDetector::SetSingleCalculationThreads(int num)
        {
            return ForwardNet::SetSingleCalculationThreads(num);
        }

        SeetaFaceInfoArray FaceDetector::Detect(const SeetaImageData& image) const
        {
            auto impl = reinterpret_cast<InnerFaceDetectorV5*>(this->m_impl);
            return impl->Detect(image);
        }

        void FaceDetector::SetMinFaceSize(int32_t size)
        {
            auto impl = reinterpret_cast<InnerFaceDetectorV5*>(this->m_impl);
            impl->m_impl->SetMinFaceSize(size);
        }

        int32_t FaceDetector::GetMinFaceSize() const
        {
            auto impl = reinterpret_cast<InnerFaceDetectorV5*>(this->m_impl);
            return impl->m_impl->GetMinFaceSize();
        }

        void FaceDetector::SetImagePyramidScaleFactor(float factor)
        {
            auto impl = reinterpret_cast<InnerFaceDetectorV5*>(this->m_impl);
            impl->m_impl->SetImagePyramidScaleFactor(factor);
        }

        float FaceDetector::GetImagePyramidScaleFactor() const
        {
            auto impl = reinterpret_cast<InnerFaceDetectorV5*>(this->m_impl);
            return impl->m_impl->GetImagePyramidScaleFactor();
        }

        void FaceDetector::SetScoreThresh(float thresh1, float thresh2, float thresh3)
        {
            auto impl = reinterpret_cast<InnerFaceDetectorV5*>(this->m_impl);
            impl->m_impl->SetScoreThresh(thresh1, thresh2, thresh3);
        }

        void FaceDetector::GetScoreThresh(float* thresh1, float* thresh2, float* thresh3) const
        {
            auto impl = reinterpret_cast<InnerFaceDetectorV5*>(this->m_impl);
            impl->m_impl->GetScoreThresh(thresh1, thresh2, thresh3);
        }

        void FaceDetector::SetVideoStable(bool stable)
        {
            auto impl = reinterpret_cast<InnerFaceDetectorV5*>(this->m_impl);
            impl->m_impl->SetVideoStable(stable);
        }

        bool FaceDetector::GetVideoStable() const
        {
            auto impl = reinterpret_cast<InnerFaceDetectorV5*>(this->m_impl);
            return impl->m_impl->GetVideoStable();
        }
    }
}
