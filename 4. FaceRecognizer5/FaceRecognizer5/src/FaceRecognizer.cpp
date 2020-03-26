#include "seeta/FaceRecognizer.h"

#include <iostream>
#include <mutex>
#include "seeta/ForwardNet.h"

#include <orz/utils/log.h>
#include <orz/io/jug/jug.h>
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include "VIPLFaceRecognizer.h"

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
        class FaceRecognizer::SharedModel
        {
        public:
            SharedModel(const SeetaModelSetting &setting)
            {
                seeta::ModelSetting exciting = setting;
                auto models = exciting.get_model();
                if (models.size() != 1)
                {
                    orz::Log(orz::FATAL) << LOG_HEAD << "Must input 1 model." << orz::crash;
                }
                orz::Log(orz::STATUS) << LOG_HEAD << "Loading models...";

                std::string model_filename = models[0];
                VIPLFaceRecognizer::Device device;
                switch (setting.device)
                {
                case SEETA_DEVICE_AUTO:
                    device = VIPLFaceRecognizer::AUTO;
                    break;
                case SEETA_DEVICE_CPU:
                    device = VIPLFaceRecognizer::CPU;
                    break;
                case SEETA_DEVICE_GPU:
                    device = VIPLFaceRecognizer::Device(VIPLFaceRecognizer::GPU0 + setting.id);
                    break;
                default:
                    device = VIPLFaceRecognizer::AUTO;
                    break;
                }
                // load model here
                m_impl.reset(new VIPLFaceRecognizerModel(model_filename.c_str(), device));
            }

            std::shared_ptr<VIPLFaceRecognizerModel> m_impl;
        };

        class FaceRecognizer::Implement
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
                VIPLFaceRecognizer::Device device;
                switch (setting.device)
                {
                case SEETA_DEVICE_AUTO:
                    device = VIPLFaceRecognizer::AUTO;
                    break;
                case SEETA_DEVICE_CPU:
                    device = VIPLFaceRecognizer::CPU;
                    break;
                case SEETA_DEVICE_GPU:
                    device = VIPLFaceRecognizer::Device(VIPLFaceRecognizer::GPU0 + setting.id);
                    break;
                default:
                    device = VIPLFaceRecognizer::AUTO;
                    break;
                }
                
                // load model here
                m_impl.reset(new VIPLFaceRecognizer(model_filename.c_str(), device));
                orz::Log(orz::STATUS) << LOG_HEAD << "Extracting " << m_impl->GetFeatureSize() << " features";
            }

            Implement(const SharedModel *model)
            {
                orz::Log(orz::STATUS) << LOG_HEAD << "Loading models...";
                // load model here
                m_impl.reset(new VIPLFaceRecognizer(*model->m_impl.get()));
                orz::Log(orz::STATUS) << LOG_HEAD << "Extracting " << m_impl->GetFeatureSize() << " features";
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

            bool ExtractCroppedFace(const SeetaImageData &image, float *features) const
            {
                if (features == nullptr) return false;
                VIPLImageData vimage(image.width, image.height, image.channels);
                vimage.data = image.data;
                bool succeed = m_impl->ExtractFeatureNormalized(vimage, features);
                return succeed;
            }

            bool Extract(const SeetaImageData &image, const SeetaPointF *points, float *features) const
            {
                if (features == nullptr) return false;
                VIPLImageData vimage(image.width, image.height, image.channels);
                vimage.data = image.data;
                VIPLPoint vpoints[5];
                for (int i = 0; i < 5; ++i)
                {
                    vpoints[i].x = points[i].x;
                    vpoints[i].y = points[i].y;
                }
                bool succeed = m_impl->ExtractFeatureWithCropNormalized(vimage, vpoints, features);
                return succeed;
            }

            float CalculateSimilarity(const float *features1, const float *features2) const
            {
                return m_impl->CalcSimilarityNormalized(
                    const_cast<float*>(features1),
                    const_cast<float*>(features2));
            }

            std::shared_ptr<VIPLFaceRecognizer> m_impl;
		};
	}
}

seeta::FaceRecognizer::FaceRecognizer(const SeetaModelSetting &setting)
	: m_impl(new Implement(setting))
{
}

seeta::FaceRecognizer::FaceRecognizer(const SharedModel* model)
    : m_impl(new Implement(model))
{
}

seeta::FaceRecognizer::~FaceRecognizer()
{
	delete m_impl;
}

int seeta::FaceRecognizer::SetLogLevel(int level)
{
	return orz::GlobalLogLevel(orz::LogLevel(level));
}

void seeta::FaceRecognizer::SetSingleCalculationThreads(int num)
{
    ForwardNet::SetSingleCalculationThreads(num);
}

const seeta::FaceRecognizer::SharedModel* seeta::FaceRecognizer::LoadModel(const SeetaModelSetting& setting)
{
    return new SharedModel(setting);
}

void seeta::FaceRecognizer::FreeModel(const SharedModel* model)
{
    delete model;
}

int seeta::FaceRecognizer::GetCropFaceWidth() const
{
    return m_impl->m_impl->GetCropWidth();
}

int seeta::FaceRecognizer::GetCropFaceHeight() const
{
    return m_impl->m_impl->GetCropHeight();
}

int seeta::FaceRecognizer::GetCropFaceChannels() const
{
    return m_impl->m_impl->GetCropChannels();
}

int seeta::FaceRecognizer::GetExtractFeatureSize() const
{
    return m_impl->m_impl->GetFeatureSize();
}

bool seeta::FaceRecognizer::CropFace(const SeetaImageData& image, const SeetaPointF* points, SeetaImageData& face) const
{
    return m_impl->CropFace(image, points, face);
}

bool seeta::FaceRecognizer::ExtractCroppedFace(const SeetaImageData& image, float* features) const
{
    return m_impl->ExtractCroppedFace(image, features);
}

bool seeta::FaceRecognizer::Extract(const SeetaImageData& image, const SeetaPointF* points, float* features) const
{
    return m_impl->Extract(image, points, features);
}

float seeta::FaceRecognizer::CalculateSimilarity(const float* features1, const float* features2) const
{
    return m_impl->CalculateSimilarity(features1, features2);
}
