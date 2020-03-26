#include "SeetaNetForwad.h"

namespace seeta
{

	NetModel::~NetModel()
	{
		this->free();
	}
	void NetModel::free()
	{
		if (buffer) SeetaFreeBuffer(buffer);
		buffer = nullptr;
		if (model) SeetaReleaseModel(model);
		model = nullptr;
		if (net) SeetaReleaseNet(net);
		net = nullptr;
	}

	bool NetModel::LoadModel(const char* model_path)
	{
		return LoadModel(model_path, SeetaDefaultDevice());
	}

	bool NetModel::LoadModel(const char* model_path, SeetaCNN_DEVICE_TYPE device)
	{
		if (!model_path)
		{
			return false;
		}
		// read file
		char *buffer;
		int64_t length;
		if (SeetaReadAllContentFromFile(model_path, &buffer, &length))
		{
			return false;
		}
		else
		{
			return LoadModel(buffer, length, device);
		}

	}

	bool NetModel::LoadModel(void* buffer, size_t length)
	{
		return LoadModel(buffer, length, SeetaDefaultDevice());
	}

	bool NetModel::LoadModel(void* buffer, size_t length, SeetaCNN_DEVICE_TYPE device)
	{
		this->free();

		this->buffer = reinterpret_cast<char *>(buffer);
		this->length = length;

		// read header
		size_t header_size = this->header.read(this->buffer, this->length);

		// convert the model
		if (SeetaReadModelFromBuffer(this->buffer + header_size, this->length - header_size, &this->model))
		{
			SeetaFreeBuffer(this->buffer);
			this->buffer = nullptr;
			return false;
		}
		SeetaFreeBuffer(this->buffer);
		this->buffer = nullptr;
		// create the net
		if (SeetaCreateNet(this->model, 1, device, &this->net))
		{
			SeetaReleaseModel(this->model);
			this->model = nullptr;
			return false;
		}
		SeetaReleaseModel(this->model);
		this->model = nullptr;

		return true;
	}

	size_t NetModel::GetFeatureSize(int number)
	{
		return this->header.feature_size * number;
	}

	size_t NetModel::GetInputWidth()
	{
		return this->header.width;
	}

	size_t NetModel::GetInputHeight()
	{
		return this->header.height;
	}

	size_t NetModel::GetInputChannels()
	{
		return this->header.channels;
	}

	bool NetModel::Valid() const
	{
		return this->net != nullptr;
	}

	static SeetaCNN_InputOutputData holiday_convert(const char* data, int width, int height, int channels)
	{
		SeetaCNN_InputOutputData input;
		input.buffer_type = SEETACNN_BGR_IMGE_CHAR;
		input.data_point_char = reinterpret_cast<unsigned char*>(const_cast<char *>(data));
		input.data_point_float = nullptr;
		input.number = 1;
		input.channel = channels;
		input.height = height;
		input.width = width;
		return input;
	}

	static SeetaCNN_InputOutputData holiday_convert(const float* data, int width, int height, int channels)
	{
		SeetaCNN_InputOutputData input;
		input.buffer_type = SEETACNN_BGR_IMGE_FLOAT;
		input.data_point_char = nullptr;
		input.data_point_float = const_cast<float *>(data);
		input.number = 1;
		input.channel = channels;
		input.height = height;
		input.width = width;
		return input;
	}

	bool NetModel::Forward(const SeetaCNN_InputOutputData* input, SeetaCNN_InputOutputData* output)
	{
		if (!this->net) return false;
		
		if (input->width != header.width || input->height != header.height || input->channel != header.channels)
		{
			return false;
		}

		int result = -1;
		if (input->data_point_char)
		{
			result = SeetaRunNetChar(this->net, 1, const_cast<SeetaCNN_InputOutputData *>(input));
		}
		else if (input->data_point_float)
		{
			result = SeetaRunNetFloat(this->net, 1, const_cast<SeetaCNN_InputOutputData *>(input));
		}

		if (result) return false;

		if (SeetaGetFeatureMap(this->net, this->header.blob_name.c_str(), output))
		{
			return false;
		}

		if (output->number * output->channel * output->height * output->width != this->header.feature_size)
		{
			return false;
		}
		
		return true;
	}

	bool NetModel::Forward(const char* data, int width, int height, int channels, SeetaCNN_InputOutputData* output)
	{
		SeetaCNN_InputOutputData input = holiday_convert(data, width, height, channels);
		return Forward(&input, output);
	}

	bool NetModel::Forward(const float* data, int width, int height, int channels, SeetaCNN_InputOutputData* output)
	{
		SeetaCNN_InputOutputData input = holiday_convert(data, width, height, channels);
		return Forward(&input, output);
	}

	static void copyData(void *to, void *from, size_t _size, size_t _count)
	{
#if _MSC_VER > 1500
		memcpy_s(to, _count * _size, from, _count * _size);
#else
		memcpy(to, from, _count * _size);
#endif
	}

	bool NetModel::Forward(const SeetaCNN_InputOutputData* input, float* feature)
	{
		SeetaCNN_InputOutputData output;
		if (!Forward(input, &output))
		{
			return  false;
		}
		copyData(feature, output.data_point_float, sizeof(float), this->header.feature_size);
		return true;
	}

	bool NetModel::Forward(const char* data, int width, int height, int channels, float* feature)
	{
		SeetaCNN_InputOutputData input = holiday_convert(data, width, height, channels);
		return Forward(&input, feature);
	}

	bool NetModel::Forward(const float* data, int width, int height, int channels, float* feature)
	{
		SeetaCNN_InputOutputData input = holiday_convert(data, width, height, channels);
		return Forward(&input, feature);
	}
}
