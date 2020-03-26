#include "HolidayForward.h"
#include "ReadFromHolidayLayer.h"
#include "HolidayNet.h"
#include "include_inner/MacroHoliday.h"
#include "HolidayCNN_proto.pb.h"
#include <openblas/cblas.h>

#ifdef HOLIDAY_GPU
#include <cuda_runtime_api.h>
#endif

struct SeetaCNN_Model
{

};

struct SeetaCNN_Net
{

};

struct SeetaCNN_SharedParam
{

};

int SeetaReadModelFromBuffer(const char* buffer, size_t buffer_length, struct SeetaCNN_Model** pmodel) {
    return HolidayCNNReadModelFromBuffer(buffer, buffer_length, (void **)pmodel);
}
int SeetaReadAllContentFromFile(const char* file_name, char** pbuffer, int64_t *file_length) {
    return ReadAllContentFromFile(file_name, pbuffer, *file_length);
}

void SeetaFreeBuffer(char *buffer)
{
	delete[] buffer;
}

int SeetaGPUDeviceCount()
{
#ifdef HOLIDAY_GPU
	int count = 0;
	if (cudaGetDeviceCount(&count) == cudaSuccess && count > 0)
		return count;
	else
		return count;
#else
	return 0;
#endif
}

int SeetaCreateNet(struct SeetaCNN_Model* model, int max_batch_size, enum SeetaCNN_DEVICE_TYPE process_device_type, struct SeetaCNN_Net** pnet) {
    return CreateNet(model, max_batch_size, process_device_type, (void **)pnet);
}

int SeetaCreateNetSharedParam(struct SeetaCNN_Model* model, int max_batch_size, enum SeetaCNN_DEVICE_TYPE process_device_type, struct SeetaCNN_Net** pnet, struct SeetaCNN_SharedParam** pparam)
{
	return CreateNetSharedParam(model, max_batch_size, process_device_type, (void **)pnet, (void **)pparam);
}

int SeetaCreateNetGPU(struct SeetaCNN_Model* model, int max_batch_size, int device_id, struct SeetaCNN_Net** pnet)
{
	return CreateNet(model, max_batch_size, HOLIDAY_CNN_GPU_DEVICE, (void **)pnet, device_id);
}

int SeetaCreateNetGPUSharedParam(struct SeetaCNN_Model* model, int max_batch_size, int device_id, struct SeetaCNN_Net** pnet, struct SeetaCNN_SharedParam** pparam)
{
	return CreateNetSharedParam(model, max_batch_size, HOLIDAY_CNN_GPU_DEVICE, (void **)pnet, (void **)pparam, device_id);
}

void SeetaNetKeepBlob(struct SeetaCNN_Net* net, const char* blob_name)
{
	HolidayKeepBlob(net, blob_name);
}

void SeetaNetKeepNoBlob(struct SeetaCNN_Net* net)
{
	HolidayKeepNoBlob(net);
}

void SeetaNetKeepAllBlob(struct SeetaCNN_Net* net)
{
	HolidayKeepAllBlob(net);
}

int SeetaNetHasKeptBlob(struct SeetaCNN_Net* net, const char* blob_name)
{
	return HolidayHasKeptBlob(net, blob_name);
}

struct SeetaCNN_SharedParam *SeetaGetSharedParam(struct SeetaCNN_Net* net)
{
	return reinterpret_cast<struct SeetaCNN_SharedParam *>(GetNetSharedParam(net));
}

int SeetaRunNetChar(struct SeetaCNN_Net* net, int counts, struct SeetaCNN_InputOutputData* pinput_data) {
    return RunNetChar(net, counts, pinput_data);
}

int SeetaRunNetFloat(struct SeetaCNN_Net* net, int counts, struct SeetaCNN_InputOutputData* pinput_data)
{
	return RunNetFloat(net, counts, pinput_data);
}

int SeetaGetFeatureMap(struct SeetaCNN_Net* net, const char* blob_name, struct SeetaCNN_InputOutputData* poutput_data) {
    return HolidayCNNGetFeatureMap(blob_name, net, poutput_data);
}

int SeetaGetAllFeatureMap(struct SeetaCNN_Net* net, int* number,struct SeetaCNN_InputOutputData** poutput_data) {
    return HolidayCNNGetAllFeatureMap( net, number,poutput_data);
}

void SeetaFreeAllFeatureMap(struct SeetaCNN_Net* net, const struct SeetaCNN_InputOutputData* poutput_data)
{
	HolidayCNNFreeAllFeatureMap(net, poutput_data);
}

void SeetaFinalizeLibrary()
{
	google::protobuf::ShutdownProtobufLibrary();
}

void SeetaReleaseNet(struct SeetaCNN_Net* net) {
    CNNReleaseNet((void **)&net);
}

void SeetaReleaseModel(struct SeetaCNN_Model* model)
{
	HolidayCNNReleaseModel((void **)&model);
}

enum SeetaCNN_DEVICE_TYPE SeetaDefaultDevice()
{
#ifdef HOLIDAY_GPU
	int count;
	if (cudaGetDeviceCount(&count) == cudaSuccess && count > 0)
		return HOLIDAY_CNN_GPU_DEVICE;
	else
		return HOLIDAY_CNN_CPU_DEVICE;
#else
	return HOLIDAY_CNN_CPU_DEVICE;
#endif
}

#define VER_HEAD(x) #x "."
#define VER_TAIL(x) #x
#define GENERATE_VER(seq) FUN_MAJOR seq
#define FUN_MAJOR(x) VER_HEAD(x) FUN_MINOR
#define FUN_MINOR(x) VER_HEAD(x) FUN_SINOR
#define FUN_SINOR(x) VER_TAIL(x)

#define SEETANET_VERSION GENERATE_VER((SEETANET_MAJOR_VERSION) (SEETANET_MINOR_VERSION) (SEETANET_SINOR_VERSION))

const char *SeetaLibraryVersionString()
{
	return SEETANET_VERSION;
}

void SeetaSetNumThreads(int num)
{
	goto_set_num_threads(num);
	openblas_set_num_threads(num);
}

int SeetaModelResetInput(struct SeetaCNN_Model* model, int width, int height)
{
	return HolidayCNNModelResetInput(model, width, height);
}

void SeetaSetNumThreadsEx(struct SeetaCNN_Net* net, int num)
{
    HolidaySetNumThreadsEx(net, num);
}
