#include "HolidayNetResource.h"
#include "HolidayBaseLayer.h"
#include "HolidayCommonCuda.h"
class HolidayPoolingGPU : public HolidayBaseLayer<float>
{
public:
	HolidayPoolingGPU();
	~HolidayPoolingGPU();
	int Process(std::vector<HolidayFeatureMap<float>*> input_data_map, std::vector<HolidayFeatureMap<float>*>& output_data_map);
	int Init(Holiday_LayerParameter& inputparam, HolidayNetResource<float> *pNetResource);
	int Exit();

	void CaculatePoolSize(int input_height, int input_width, int &output_height, int &output_width);

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

	std::string m_tf_padding;
	int m_tf_fake_padding_h = 0;
	int m_tf_fake_padding_w = 0;
};