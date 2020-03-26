#include "seeta/FaceComparer.h"

#include "VIPLFaceComparer.h"

#include <orz/utils/log.h>

#define VER_HEAD(x) #x "."
#define VER_TAIL(x) #x
#define GENERATE_VER(seq) FUN_MAJOR seq
#define FUN_MAJOR(x) VER_HEAD(x) FUN_MINOR
#define FUN_MINOR(x) VER_HEAD(x) FUN_SINOR
#define FUN_SINOR(x) VER_TAIL(x)

#define LIBRARY_VERSION GENERATE_VER( \
	(SEETA_FACE_RECOGNIZER_MAJOR_VERSION) \
	(SEETA_FACE_RECOGNIZER_MINOR_VERSION) \
	(SEETA_FACE_RECOGNIZER_SINOR_VERSION))

#define LIBRARY_NAME "FaceRecognizer"

#define LOG_HEAD LIBRARY_NAME "(" LIBRARY_VERSION "): "

namespace seeta
{
	namespace v500
	{
        class FaceComparer::Implement
		{
		public:
            using self = Implement;

            Implement(const SeetaModelSetting &setting)
			{
				seeta::ModelSetting exciting = setting;
				auto models = exciting.get_model();
				if (models.size() != 1)
				{
                    orz::Log(orz::FATAL) << LOG_HEAD << "Must input 1 model." << orz::crash;
				}
                orz::Log(orz::STATUS) << LOG_HEAD << "Loading models...";

                std::string model_filename = models[0];
                
                // load model here
                m_impl.reset(new VIPLFaceComparer(model_filename.c_str()));
            }

            bool CropFace(const SeetaImageData &image, const SeetaPointF *points, SeetaImageData &face) const
            {
                if (points == nullptr) return false;
                VIPLImageData vimage(image.width, image.height, image.channels);
                vimage.data = image.data;
                VIPLPoint vpoints[5];
                for (int i = 0; i < 5; ++i)
                {
                    vpoints[i].x = points[i].x;
                    vpoints[i].y = points[i].y;
                }
                VIPLImageData vface(face.width, face.height, face.channels);
                vface.data = face.data;
                bool succeed = m_impl->CropFace(vimage, vpoints, vface);
                return succeed;
            }

            float CalculateSimilarity(const float *features1, const float *features2) const
            {
                return m_impl->CalcSimilarityNormalized(
                    const_cast<float*>(features1),
                    const_cast<float*>(features2));
            }

            std::shared_ptr<VIPLFaceComparer> m_impl;
		};
	}
}

seeta::FaceComparer::FaceComparer(const SeetaModelSetting &setting)
	: m_impl(new Implement(setting))
{
}

seeta::FaceComparer::~FaceComparer()
{
	delete m_impl;
}

int seeta::FaceComparer::GetCropFaceWidth() const
{
    return m_impl->m_impl->GetCropWidth();
}

int seeta::FaceComparer::GetCropFaceHeight() const
{
    return m_impl->m_impl->GetCropHeight();
}

int seeta::FaceComparer::GetCropFaceChannels() const
{
    return m_impl->m_impl->GetCropChannels();
}

bool seeta::FaceComparer::CropFace(const SeetaImageData& image, const SeetaPointF* points, SeetaImageData& face) const
{
    return m_impl->CropFace(image, points, face);
}

float seeta::FaceComparer::CalculateSimilarity(const float* features1, const float* features2) const
{
    return m_impl->CalculateSimilarity(features1, features2);
}
