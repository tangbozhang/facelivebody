#ifndef _SEETA_NET_FORWARD_H
#define _SEETA_NET_FORWARD_H

#include "SeetaModelHeader.h"
#include "HolidayForward.h"

namespace seeta
{
	class NetModel
	{
	public:
		~NetModel();
	private:
		char *buffer = nullptr;
		int64_t length;
		SeetaCNN_Model *model = nullptr;
		SeetaCNN_Net *net = nullptr;
		ModelHeader header;
	public:

		void free();

		bool LoadModel(const char *model_path);
		bool LoadModel(const char *model_path, SeetaCNN_DEVICE_TYPE device);
		bool LoadModel(void *buffer, size_t length);
		bool LoadModel(void *buffer, size_t length, SeetaCNN_DEVICE_TYPE device);

		size_t GetFeatureSize(int number = 1);
		size_t GetInputWidth();
		size_t GetInputHeight();
		size_t GetInputChannels();
		bool Valid() const;

		bool Forward(const SeetaCNN_InputOutputData *input, SeetaCNN_InputOutputData *output);
		bool Forward(const char *data, int width, int height, int channels, SeetaCNN_InputOutputData *output);
		bool Forward(const float *data, int width, int height, int channels, SeetaCNN_InputOutputData *output);
		bool Forward(const SeetaCNN_InputOutputData *input, float *feature);
		bool Forward(const char *data, int width, int height, int channels, float *feature);
		bool Forward(const float *data, int width, int height, int channels, float *feature);
		bool Forward(const unsigned char *data, int width, int height, int channels, float *feature)
		{
			return Forward(reinterpret_cast<const char *>(data), width, height, channels, feature);
		}
	};
}

#endif // _SEETA_NET_FORWARD_H
