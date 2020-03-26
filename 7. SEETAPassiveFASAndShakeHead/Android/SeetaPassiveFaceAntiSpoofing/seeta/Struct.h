#ifndef INC_SEETA_STRUCT_H
#define INC_SEETA_STRUCT_H

#include "CStruct.h"

#include <memory>
#include <algorithm>
#include <vector>
#include <cstring>
#include <string>

#define INCLUDED_SEETA_STRUCT

namespace seeta
{
	class ImageData : public SeetaImageData
	{
	public:
		using self = ImageData;
		using supper = SeetaImageData;

		using byte = unsigned char;

		ImageData(const supper &image)
			: ImageData(image.data, image.width, image.height, image.channels) {}

		ImageData(int width, int height, int channels)
		{
			supper::width = width;
			supper::height = height;
			supper::channels = channels;
			self::m_data.reset(new byte[self::count()], std::default_delete<byte[]>());
			supper::data = self::m_data.get();
		}

		ImageData() : ImageData(0, 0, 0) {}

		ImageData(const byte *data, int width, int height, int channels)
			: ImageData(width, height, channels)
		{
			self::copy_from(data);
		}

		void copy_from(const byte *data, int size = -1)
		{
			int copy_size = this->count();
			copy_size = size < 0 ? copy_size : std::min<int>(copy_size, size);
			copy(self::data, data, copy_size);
		}

		void copy_to(byte *data, int size = -1) const
		{
			int copy_size = this->count();
			copy_size = size < 0 ? copy_size : std::min<int>(copy_size, size);
			copy(data, self::data, copy_size);
		}

		static void copy(byte *dst, const byte *src, size_t size)
		{
#if _MSC_VER >= 1600
			memcpy_s(dst, size, src, size);
#else
			memcpy(dst, src, size);
#endif
		}

		int count() const { return self::width * self::height * self::channels; }

		ImageData clone() const
		{
			return ImageData(self::data, self::width, self::height, self::channels);
		}

	private:
		std::shared_ptr<byte> m_data;
	};

	class Point : public SeetaPoint
	{
	public:
		using self = Point;
		using supper = SeetaPoint;

		Point(const supper &other) : supper(other) {}

		Point(int x, int y)
		{
			supper::x = x;
			supper::y = y;
		}

		Point() : Point(0, 0) {}
	};

	class PointF : public SeetaPointF
	{
	public:
		using self = PointF;
		using supper = SeetaPointF;

		PointF(const supper &other) : supper(other) {}

		PointF(double x, double y)
		{
			supper::x = x;
			supper::y = y;
		}

		PointF() : PointF(0, 0) {}
	};

	class Size : public SeetaSize
	{
	public:
		using self = Size;
		using supper = SeetaSize;

		Size(const supper &other) : supper(other) {}

		Size(int width, int height)
		{
			supper::width = width;
			supper::height = height;
		}

		Size() : Size(0, 0) {}
	};

	class Rect : public SeetaRect
	{
	public:
		using self = Rect;
		using supper = SeetaRect;

		Rect(const supper &other) : supper(other) {}

		Rect(int x, int y, int width, int height)
		{
			supper::x = x;
			supper::y = y;
			supper::width = width;
			supper::height = height;
		}

		Rect() : Rect(0, 0, 0, 0) {}

		Rect(int x, int y, const Rect &rect)
		{
			supper::x = x;
			supper::y = y;
			supper::width = rect.width;
			supper::height = rect.height;
		}

		Rect(const Point &point, int width, int height)
		{
			supper::x = point.x;
			supper::y = point.y;
			supper::width = width;
			supper::height = height;
		}

		Rect(const Point &point, const Rect &rect)
		{
			supper::x = point.x;
			supper::y = point.y;
			supper::width = rect.width;
			supper::height = rect.height;
		}

		operator Point() const { return Point(self::x, self::y); }

		operator Size() const { return Size(self::width, self::height); }
	};

	class Region : public SeetaRegion
	{
	public:
		using self = Region;
		using supper = SeetaRegion;

		Region(const supper &other) : supper(other) {}

		Region(int top, int bottom, int left, int right)
		{
			supper::top = top;
			supper::bottom = bottom;
			supper::left = left;
			supper::right = right;
		}

		Region() : Region(0, 0, 0, 0) {}

		Region(const Rect &rect) : Region(rect.y, rect.y + rect.height, rect.x, rect.x + rect.width) {}

		operator Rect() const { return Rect(left, top, right - left, bottom - top); }
	};

	class ModelSetting : public SeetaModelSetting
	{
	public:
		using self = ModelSetting;
		using supper = SeetaModelSetting;

		enum Device
		{
			AUTO,
			CPU,
			GPU
		};

		ModelSetting()
		{
			supper::device = SeetaDevice(AUTO);
			supper::id = 0;
			self::update();
		}

		ModelSetting(const supper &other)
		{
			supper::device = other.device;
			supper::id = other.id;
			int i = 0;
			while (other.model[i])
			{
				m_model_string.push_back(other.model[i]);
				++i;
			}
			self::update();
		}

		ModelSetting(const std::string &model, Device device = AUTO, int id = 0)
		{
			supper::device = SeetaDevice(device);
			supper::id = id;
			append(model);
		}

		ModelSetting(const std::vector<std::string> &model, Device device = AUTO, int id = 0)
		{
			supper::device = SeetaDevice(device);
			supper::id = id;
			append(model);
		}

		ModelSetting(Device device, int id = 0)
		{
			supper::device = SeetaDevice(device);
			supper::id = id;
			self::update();
		}

		Device get_device() const { return Device(device); }
		int get_id() const { return id; }

		Device set_device(Device device)
		{
			auto old = supper::device;
			supper::device = SeetaDevice(device);
			return Device(old);
		}

		int set_id(int id)
		{
			auto old = supper::id;
			supper::id = id;
			return old;
		}

		void clear()
		{
			m_model_string.clear();
			self::update();
		}

		void append(const std::string &model)
		{
			m_model_string.push_back(model);
			self::update();
		}

		void append(const std::vector<std::string> &model)
		{
			m_model_string.insert(m_model_string.end(), model.begin(), model.end());
			self::update();
		}

		const std::vector<std::string> &get_model() const
		{
			return m_model_string;
		}

	private:
		std::vector<const char *> m_model;
		std::vector<std::string> m_model_string;

		/**
		 * \brief build supper::model
		 */
		void update()
		{
			m_model.clear();
			m_model.reserve(m_model_string.size() + 1);
			for (auto &model_string : m_model_string)
			{
				m_model.push_back(model_string.c_str());
			}
			m_model.push_back(nullptr);
			supper::model = m_model.data();
		}
	};
}

#endif
