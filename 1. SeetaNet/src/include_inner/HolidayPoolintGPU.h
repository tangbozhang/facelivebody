#include "HolidayNetResource.h"
#include "HolidayBaseLayer.h"
#include "HolidayCommonCuda.h"
class Pooling_gpu : public HolidayBaseLayer<float>
{
public:
	Pooling_gpu();
	~Pooling_gpu();
	int Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map);
	int Init(Holiday_LayerParameter& inputparam, HolidayNetResource<float> *pNetResource);
	int GetTopSize(std::vector<HolidayDataSize>& out_data_size);
	int Exit();
private:
	HolidayNetResourceGpu *pNetResourceGpu;
	int dwType;
	int dwKernelH;
	int dwKernelW;
	int dwStrideH;
	int dwStrideW;
	int dwPadH;
	int dwPadW;

	bool dwValid;
};