#include "VIPLFaceCrop.h"

#include "common_alignment.h"
#include <memory>

bool VIPL::FaceMeanShape(VIPLPoint* mean_shape, int num, int* mean_shape_size, int id)
{
	if (num != 5 || (id != 0 && id != 1))
	{
		return false;
	}

	if (id == 0)
	{
		mean_shape[0] = { 89.3095, 72.9025 };
		mean_shape[1] = { 169.3095, 72.9025 };
		mean_shape[2] = { 127.8949, 127.0441 };
		mean_shape[3] = { 96.8796, 184.8907 };
		mean_shape[4] = { 159.1065, 184.7601 };
		*mean_shape_size = 256;
	}
	else
	{
		mean_shape[0] = { 89.3095, 102.9025 };
		mean_shape[1] = { 169.3095, 102.9025 };
		mean_shape[2] = { 127.8949, 157.0441 };
		mean_shape[3] = { 96.8796, 214.8907 };
		mean_shape[4] = { 159.1065, 214.7601 };
		*mean_shape_size = 256;
	}

	return true;
}

void VIPL::ResizeMeanShape(VIPLPoint* mean_shape, int num, double scaler)
{
	for (int i = 0; i < num; ++i)
	{
		mean_shape[i].x *= scaler;
		mean_shape[i].y *= scaler;
	}
}

bool VIPL::FaceCrop(
	const VIPLImageData& src_img,
	VIPLImageData& dst_img,
	const VIPLPoint* points, int num,
	const VIPLPoint* mean_shape, int mean_shape_size,
	CROP_METHOD method,
	VIPLPoint* final_points,
	int final_size)
{
	if (final_size < 0) final_size = mean_shape_size;
	if (dst_img.width != final_size || dst_img.height != final_size || dst_img.channels != src_img.channels) return false;


	std::unique_ptr<float[]> crop_points(new float[num * 2]);
	std::unique_ptr<float[]> crop_mean_shape(new float[num * 2]);
	std::unique_ptr<float[]> crop_final_points;
	SAMPLING_TYPE type;
	for (int i = 0; i < num; ++i)
	{
		crop_points[i * 2] = static_cast<float>(points[i].x);
		crop_points[i * 2 + 1] = static_cast<float>(points[i].y);
		crop_mean_shape[i * 2] = static_cast<float>(mean_shape[i].x);
		crop_mean_shape[i * 2 + 1] = static_cast<float>(mean_shape[i].y);
	}
	if (final_points)
	{
		crop_final_points.reset(new float[num * 2]);
	}
	switch (method)
	{
	default:
		type = LINEAR;
		break;
	case BY_LINEAR:
		type = LINEAR;
		break;
	case BY_BICUBIC:
		type = BICUBIC;
		break;
	}

	bool success = face_crop_core(
		src_img.data, src_img.width, src_img.height, src_img.channels,
		dst_img.data, mean_shape_size, mean_shape_size,
		crop_points.get(), num,
		crop_mean_shape.get(), mean_shape_size, mean_shape_size,
		(final_size - mean_shape_size) / 2,
		(final_size - mean_shape_size) - (final_size - mean_shape_size) / 2,
		(final_size - mean_shape_size) / 2,
		(final_size - mean_shape_size) - (final_size - mean_shape_size) / 2,
		final_points ? crop_final_points.get() : nullptr,
		type);

	if (final_points && success)
	{
		for (int i = 0; i < num; ++i)
		{
			final_points[i].x = crop_final_points.get()[2 * i];
			final_points[i].y = crop_final_points.get()[2 * i + 1];
		}
	}
	return success;
}
