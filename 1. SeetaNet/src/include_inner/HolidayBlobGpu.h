#ifndef HOLIDAY_BLOb_GPU_H__
#define HOLIDAY_BLOb_GPU_H__

#include <vector>
class HolidayBlobGpu
{
public:
	HolidayBlobGpu();

	~HolidayBlobGpu();
	

	int n_max_num;
	char *pbyData_cpu;
	float *pfData_cpu;
	float *pfData_gpu;
	int Gpu_init(void *pNetResourceGpu);
	int Gpu_DataIn(void *pNetResourceGpu, int dwStorageType, void *pDataIn);
	int Gpu_DataOut(void *pNetResourceGpu, int dwStorageType, float *out);
	int Gpu_free();

	std::vector<int> shape_;

	int data_size;
	int memory_size;

private:

};




#endif