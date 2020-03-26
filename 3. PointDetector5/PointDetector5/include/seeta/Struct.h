#ifndef INC_SEETA_STRUCT_H
#define INC_SEETA_STRUCT_H

#include "CStruct.h"

#include <memory>
#include <algorithm>
#include <vector>
#include <cstring>
#include <istream>

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
			this->width = width;
			this->height = height;
			this->channels = channels;
			this->m_data.reset(new byte[this->count()], std::default_delete<byte[]>());
			this->data = this->m_data.get();
		}

		ImageData() : ImageData(0, 0, 0) {}

		ImageData(const byte *data, int width, int height, int channels)
			: ImageData(width, height, channels)
		{
			this->copy_from(data);
		}

		void copy_from(const byte *data, int size = -1)
		{
			int copy_size = this->count();
			copy_size = size < 0 ? copy_size : std::min<int>(copy_size, size);
			this->copy(this->data, data, copy_size);
		}

		void copy_to(byte *data, int size = -1) const
		{
			int copy_size = this->count();
			copy_size = size < 0 ? copy_size : std::min<int>(copy_size, size);
			this->copy(data, this->data, copy_size);
		}

		static void copy(byte *dst, const byte *src, size_t size)
		{
			std::memcpy(dst, src, size);
		}

		int count() const { return this->width * this->height * this->channels; }

		ImageData clone() const
		{
			return ImageData(this->data, this->width, this->height, this->channels);
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
			this->x = x;
			this->y = y;
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
			this->x = x;
			this->y = y;
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
			this->width = width;
			this->height = height;
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
			this->x = x;
			this->y = y;
			this->width = width;
			this->height = height;
		}

		Rect() : Rect(0, 0, 0, 0) {}

		Rect(int x, int y, const Rect &rect)
		{
			this->x = x;
			this->y = y;
			this->width = rect.width;
			this->height = rect.height;
		}

		Rect(const Point &point, int width, int height)
		{
			this->x = point.x;
			this->y = point.y;
			this->width = width;
			this->height = height;
		}

		Rect(const Point &point, const Rect &rect)
		{
			this->x = point.x;
			this->y = point.y;
			this->width = rect.width;
			this->height = rect.height;
		}

		operator Point() const { return Point(this->x, this->y); }

		operator Size() const { return Size(this->width, this->height); }
	};

	class Region : public SeetaRegion
	{
	public:
		using self = Region;
		using supper = SeetaRegion;

		Region(const supper &other) : supper(other) {}

		Region(int top, int bottom, int left, int right)
		{
			this->top = top;
			this->bottom = bottom;
			this->left = left;
			this->right = right;
		}

		Region() : Region(0, 0, 0, 0) {}

		Region(const Rect &rect) : Region(rect.y, rect.y + rect.height, rect.x, rect.x + rect.width) {}

		operator Rect() const { return Rect(left, right - left, top, bottom - top); }
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
			this->device = SeetaDevice(AUTO);
			this->id = 0;
			this->update();
		}

		ModelSetting(const supper &other)
		{
			this->device = other.device;
			this->id = other.id;
			int i = 0;
			while (other.model[i])
			{
				m_model_string.push_back(other.model[i]);
				++i;
			}
			this->update();
		}

		ModelSetting(const std::string &model, Device device = AUTO, int id = 0)
		{
			this->device = SeetaDevice(device);
			this->id = id;
			this->append(model);
		}

		ModelSetting(const std::vector<std::string> &model, Device device = AUTO, int id = 0)
		{
			this->device = SeetaDevice(device);
			this->id = id;
			this->append(model);
		}

		ModelSetting(Device device, int id = 0)
		{
			this->device = SeetaDevice(device);
			this->id = id;
			this->update();
		}

		Device get_device() const { return Device(this->device); }
		int get_id() const { return this->id; }

		Device set_device(Device device)
		{
			auto old = this->device;
			this->device = SeetaDevice(device);
			return Device(old);
		}

		int set_id(int id)
		{
			auto old = this->id;
			this->id = id;
			return old;
		}

		void clear()
		{
			this->m_model_string.clear();
			this->update();
		}

		void append(const std::string &model)
		{
			this->m_model_string.push_back(model);
			this->update();
		}

		void append(const std::vector<std::string> &model)
		{
			this->m_model_string.insert(this->m_model_string.end(), model.begin(), model.end());
			this->update();
		}

		const std::vector<std::string> &get_model() const
		{
			return this->m_model_string;
		}

		const std::string &get_model(size_t i) const
		{
			return this->m_model_string[i];
		}

	private:
		std::vector<const char *> m_model;
		std::vector<std::string> m_model_string;

		/**
		 * \brief build buffer::model
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
			this->model = m_model.data();
		}
	};

	class Buffer : public SeetaBuffer
	{
	public:
		using self = Buffer;
		using supper = SeetaBuffer;

		using byte = unsigned char;

		Buffer(const supper &other) : Buffer(other.buffer, other.size) {}

		Buffer(const void *buffer, int64_t size)
		{
			this->m_size = size;
			if (this->m_size)
			{
				this->m_buffer.reset(new byte[size_t(this->m_size)], std::default_delete<byte[]>());
			}
			this->buffer = this->m_buffer.get();
			this->size = this->m_size;
			this->copy_from(buffer, size);
		}

		explicit Buffer(int64_t size) : Buffer(nullptr, size) {}

		Buffer() : Buffer(nullptr, 0) {}

		Buffer(std::istream &in, int64_t size = -1)
		{
			if (size < 0)
			{
				auto cur = in.tellg();
				in.seekg(0, std::ios::end);
				auto end = in.tellg();
				size = int64_t(end - cur);
				in.seekg(-size, std::ios::end);
			}

			this->m_size = size;
			this->m_buffer.reset(new byte[size_t(this->m_size)], std::default_delete<byte[]>());
			this->buffer = this->m_buffer.get();
			this->size = this->m_size;

			in.read(reinterpret_cast<char*>(this->buffer), this->size);
		}

		void copy_from(const void *data, int64_t size = -1)
		{
			auto copy_size = this->m_size;
			copy_size = size < 0 ? copy_size : std::min<int64_t>(copy_size, size);
			copy(this->buffer, data, size_t(copy_size));
		}

		void copy_to(byte *data, int64_t size = -1) const
		{
			auto copy_size = this->m_size;
			copy_size = size < 0 ? copy_size : std::min<int64_t>(copy_size, size);
			copy(data, this->buffer, size_t(copy_size));
		}

		static void copy(void *dst, const void *src, size_t size)
		{
			if (dst == nullptr || src == nullptr) return;
			std::memcpy(dst, src, size);
		}

		Buffer clone() const
		{
			return self(this->buffer, this->size);
		}

		void rebind(const void *buffer, int64_t size)
		{
			if (size < 0) size = 0;
			if (size > this->m_size)
			{
				this->m_buffer.reset(new byte[size_t(size)], std::default_delete<byte[]>());
			}
			this->m_size = size;
			this->buffer = this->m_buffer.get();
			this->size = this->m_size;
			this->copy_from(buffer, size);
		}

	private:
		std::shared_ptr<byte> m_buffer;
		int64_t m_size = 0;
	};

	class ModelBuffer : SeetaModelBuffer
	{
	public:
		using self = ModelBuffer;
		using supper = SeetaModelBuffer;


		enum Device
		{
			AUTO = SEETA_DEVICE_AUTO,
			CPU  = SEETA_DEVICE_CPU,
			GPU  = SEETA_DEVICE_GPU,
		};

		ModelBuffer()
		{
			this->device = SeetaDevice(AUTO);
			this->id = 0;
			this->update();
		}

		ModelBuffer(const supper &other)
		{
			this->device = other.device;
			this->id = other.id;
			int i = 0;
			while (other.buffer[i].buffer && other.buffer[i].size)
			{
				m_model_buffer.push_back(other.buffer[i]);
				++i;
			}
			this->update();
		}

		ModelBuffer(const seeta::Buffer &buffer, Device device = AUTO, int id = 0)
		{
			this->device = SeetaDevice(device);
			this->id = id;
			this->append(buffer);
		}

		ModelBuffer(const std::vector<seeta::Buffer> &buffer, Device device = AUTO, int id = 0)
		{
			this->device = SeetaDevice(device);
			this->id = id;
			this->append(buffer);
		}

		explicit ModelBuffer(Device device, int id = 0)
		{
			this->device = SeetaDevice(device);
			this->id = id;
			this->update();
		}

		Device get_device() const { return Device(this->device); }
		int get_id() const { return this->id; }

		Device set_device(Device device)
		{
			auto old = this->device;
			this->device = SeetaDevice(device);
			return Device(old);
		}

		int set_id(int id)
		{
			auto old = this->id;
			this->id = id;
			return old;
		}

		void clear()
		{
			this->m_model_buffer.clear();
			this->update();
		}

		void append(const seeta::Buffer &buffer)
		{
			this->m_model_buffer.push_back(buffer);
			this->update();
		}

		void append(const std::vector<seeta::Buffer> &model)
		{
			this->m_model_buffer.insert(this->m_model_buffer.end(), model.begin(), model.end());
			this->update();
		}

		const std::vector<seeta::Buffer> &get_buffer() const
		{
			return this->m_model_buffer;
		}

		const seeta::Buffer &get_buffer(size_t i) const
		{
			return this->m_model_buffer[i];
		}

	private:
		std::vector<SeetaBuffer> m_buffer;
		std::vector<seeta::Buffer> m_model_buffer;

		/**
		 * \brief build supper::buffer
		 */
		void update()
		{
			this->m_buffer.clear();
			this->m_buffer.reserve(m_model_buffer.size() + 1);
			for (auto &model_buffer : m_model_buffer)
			{
				this->m_buffer.push_back(model_buffer);
			}
			this->m_buffer.push_back(seeta::Buffer());	// terminate with empty buffer
			this->buffer = this->m_buffer.data();
		}
	};
}

#endif
