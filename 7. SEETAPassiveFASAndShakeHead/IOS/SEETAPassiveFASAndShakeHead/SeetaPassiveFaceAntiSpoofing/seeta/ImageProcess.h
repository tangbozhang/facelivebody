#pragma once

#include "DataHelper.h"
#include "Struct.h"
#include "CommonStruct.h"


namespace seeta
{
	const Image color(const Image &img);
	const Image gray(const Image &img);
	const Image crop(const Image &img, const Rect &rect);
	using Padding = Size;
	const Image pad(const Image &img, const Padding& padding);
	const Image resize(const Image &img, const Size &size);
	const Image crop_resize(const Image &img, const Rect &rect, const Size &size);
	const Image equalize_hist(const Image &img);

	void fill(Image &img, const Point &point, const Image &patch);
	void fill(Image &img, const Rect &rect, const Image &patch);
	
}