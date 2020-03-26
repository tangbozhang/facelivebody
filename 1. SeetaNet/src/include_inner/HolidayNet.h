#ifndef _HOLIDAY_H__
#define _HOLIDAY_H__

#include "HolidayStruct.h"

#define NetF float


enum MyEnum
{
    BLOB_NAME_NOT_EXIST = -1,
};


/**
 * \brief 
 * \param model 
 * \param max_batch_size 
 * \param process_device_type 
 * \param output_net_out 
 * \param gpu_device_id only process_device_type is HOLIDAY_CNN_GPU_DEVICE, this parameter works
 * \return 
 */
int CreateNet(void* model, int max_batch_size, SeetaCNN_DEVICE_TYPE process_device_type, void** output_net_out, int gpu_device_id = 0);

/**
 * @note only process_device_type is HOLIDAY_CNN_GPU_DEVICE, this parameter works
 */
int CreateNetSharedParam(void* model, int max_batchsize, SeetaCNN_DEVICE_TYPE process_device_type, void** output_net_out, void** output_shared_param, int gpu_device_id = 0);

int RunNetChar(void* output_net_out, int counts, SeetaCNN_InputOutputData* pinput_Data);
int RunNetFloat(void* output_net_out, int counts, SeetaCNN_InputOutputData* pinput_Data);
int HolidayCNNGetFeatureMap(const char* buffer_name, void *pNetIn, SeetaCNN_InputOutputData* outputData);

int HolidayCNNGetAllFeatureMap(void *pNetIn, int* number, SeetaCNN_InputOutputData** outputData);
void HolidayCNNFreeAllFeatureMap(void * pNetIn, const SeetaCNN_InputOutputData *outputData);

void *GetNetSharedParam(void *net);

void CNNReleaseNet(void** pNetIn);

int HolidayCNNReleaseSharedParam(void** shared_param);

void HolidayKeepBlob(struct SeetaCNN_Net* net, const char *blob_name);

void HolidayKeepNoBlob(struct SeetaCNN_Net *net);

void HolidayKeepAllBlob(struct SeetaCNN_Net *net);

int HolidayHasKeptBlob(struct SeetaCNN_Net* net, const char *blob_name);

void HolidaySetNumThreadsEx(struct SeetaCNN_Net* net, int num);

#endif