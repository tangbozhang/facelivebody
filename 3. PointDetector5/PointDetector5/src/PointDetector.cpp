#include "seeta/PointDetector.h"

#include <iostream>
#include <mutex>

#include <orz/utils/log.h>
#include <orz/io/jug/jug.h>
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>

#include "VIPLPointDetector.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define VER_HEAD(x) #x "."
#define VER_TAIL(x) #x
#define GENERATE_VER(seq) FUN_MAJOR seq
#define FUN_MAJOR(x) VER_HEAD(x) FUN_MINOR
#define FUN_MINOR(x) VER_HEAD(x) FUN_SINOR
#define FUN_SINOR(x) VER_TAIL(x)

#define LIBRARY_VERSION GENERATE_VER( \
	(SEETA_POINT_DETECTOR_MAJOR_VERSION) \
	(SEETA_POINT_DETECTOR_MINOR_VERSION) \
	(SEETA_POINT_DETECTOR_SINOR_VERSION))

#define LIBRARY_NAME "PointDetector"

#define LOG_HEAD LIBRARY_NAME "(" LIBRARY_VERSION "): "

//#define WITH_OPENCV_DEBUG
#ifdef WITH_OPENCV_DEBUG
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif

static std::vector<int> version(const std::string &v)
{
    (void)(&version);
	std::vector<int> vi;
	auto vs = orz::Split(v, '.');
	for (auto &s : vs) vi.push_back(std::atoi(s.c_str()));
	while (vi.size() < 3) vi.push_back(0);
	return std::move(vi);
}

namespace seeta
{
	namespace v500
	{
		class InnerPointDetector
		{
		public:
			using self = InnerPointDetector;

			InnerPointDetector(const SeetaModelSetting &setting)
			{
				std::vector<std::string> model;
				int index = 0;
				while (setting.model[index]) model.push_back(setting.model[index++]);
				if (model.empty())
				{
                    orz::Log(orz::FATAL) << LOG_HEAD << "Must input 1 model." << orz::crash;
                }
                orz::Log(orz::STATUS) << LOG_HEAD << "Loading models...";
                m_impl.reset(new VIPLPointDetector(model[0].c_str()));
                orz::Log(orz::INFO) << LOG_HEAD << "Using device " << "CPU";
                orz::Log(orz::STATUS) << LOG_HEAD << "Detecting " << LandmarkNum() << " landmarks";
			}

			int LandmarkNum() const
			{
				return m_impl->LandmarkNum();
			}

			bool DetectV5(const SeetaImageData &image, const SeetaRect &face, SeetaPointF *points, int *masks = nullptr) const
			{
				if (points == nullptr) return false;
				VIPLImageData vimage(image.width, image.height, image.channels);
				vimage.data = image.data;
				std::vector<VIPLPoint> vpoints;
				std::vector<int> vmasks;
				VIPLFaceInfo vface;
				vface.x = face.x;
				vface.y = face.y;
				vface.width = face.width;
				vface.height = face.height;
				bool succeed = m_impl->DetectLandmarks(vimage, vface, vpoints);
				if (!succeed) return false;
				for (size_t i = 0; i < vpoints.size(); ++i)
				{
					points[i].x = vpoints[i].x;
					points[i].y = vpoints[i].y;
				}
                if (masks)
                {
                    for (size_t i = 0; i < vmasks.size(); ++i)
                    {
                        masks[i] = vmasks[i];
                    }
                }
				return true;
			}

		private:
			std::shared_ptr<VIPLPointDetector> m_impl;
		};
	}
}

// Header
#ifdef NEED_CHECK
#include "encryption/code/checkit.h"
#endif

seeta::PointDetector::PointDetector(const SeetaModelSetting &setting)
	: m_impl(new InnerPointDetector(setting))
{

// Code
#ifdef NEED_CHECK
	checkit();
#endif
}

seeta::PointDetector::~PointDetector()
{
	auto impl = reinterpret_cast<InnerPointDetector*>(this->m_impl);
	delete impl;
}

int seeta::PointDetector::SetLogLevel(int level)
{
    return orz::GlobalLogLevel(orz::LogLevel(level));
}

int seeta::PointDetector::GetLandmarkNumber() const
{
	auto impl = reinterpret_cast<InnerPointDetector*>(this->m_impl);
	return impl->LandmarkNum();
}

bool seeta::PointDetector::Detect(const SeetaImageData& image, const SeetaRect& face,
                                  SeetaPointF* points) const
{
	auto impl = reinterpret_cast<InnerPointDetector*>(this->m_impl);
	return impl->DetectV5(image, face, points);
}

bool seeta::PointDetector::Detect(const SeetaImageData& image, const SeetaRect& face, SeetaPointF* points,
    int* masks) const
{
    auto impl = reinterpret_cast<InnerPointDetector*>(this->m_impl);
    return impl->DetectV5(image, face, points, masks);
}
