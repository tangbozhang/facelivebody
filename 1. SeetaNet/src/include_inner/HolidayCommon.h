#ifndef _HOLIDAY_COMMON_H__
#define _HOLIDAY_COMMON_H__


struct HolidayDataSize
{
	std::vector<int> data_dim;
	HolidayDataSize()
	{
		data_dim.clear();
	};

	HolidayDataSize(const HolidayDataSize& a)
	{
		data_dim = a.data_dim;
	};
};

#endif