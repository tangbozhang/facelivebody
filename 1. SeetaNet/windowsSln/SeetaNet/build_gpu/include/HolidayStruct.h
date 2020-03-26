#ifndef _HOLIDAY_STRUCT_H
#define _HOLIDAY_STRUCT_H

#if defined(SEETA_EXPORTS)
#define HOLIDAY_EXPORTS
#endif

#if defined(_MSC_VER)
	#ifdef HOLIDAY_EXPORTS
		#define HOLIDAY_API __declspec(dllexport)
	#else
		#define HOLIDAY_API __declspec(dllimport)
	#endif
#else
	#define HOLIDAY_API
#endif

#ifdef __cplusplus
#define HOLIDAY_C_API extern "C" HOLIDAY_API
#else
#define HOLIDAY_C_API HOLIDAY_API
#endif

/**
* @brief The supported device.
*/
enum SeetaCNN_DEVICE_TYPE
{
	HOLIDAY_CNN_CPU_DEVICE = 0,		/**< CPU, default */
	HOLIDAY_CNN_GPU_DEVICE = 1,		/**< GPU, only supported in gpu version */
	HOLIDAY_CNN_OPENCL_DEVICE = 2	/**< OPENCL, not recommend */
};
typedef enum SeetaCNN_DEVICE_TYPE SeetaCNN_DEVICE_TYPE;

/**
* @brief The dummy model structure
*/
struct SeetaCNN_Model;
typedef struct SeetaCNN_Model SeetaCNN_Model;

/**
* @brief The dummy net structure
*/
struct SeetaCNN_Net;
typedef struct SeetaCNN_Net SeetaCNN_Net;
                              

/**
* @brief The dummy SharedParam structure
*/
struct SeetaCNN_SharedParam;
typedef struct SeetaCNN_SharedParam SeetaCNN_SharedParam;

//for buffer_type enum
typedef enum
{
	SEETACNN_BGR_IMGE_CHAR = 0,
	SEETACNN_BGR_IMGE_FLOAT = 1,
	SEETACNN_NCHW_FLOAT = 2,

}SEETACNN_BUFFER_STORAGE_ODER_TYPE;

/**
* @brief The base data structure
*/
struct SeetaCNN_InputOutputData
{
	float* data_point_float;	/**< Used in output mode, pointing to the specific blob */
	unsigned char* data_point_char;		/**< Used in input mode, pointing to image data */
	int number;					/**< Number of the batch size */
	int channel;				/**< Number of the channels */
	int width;					/**< Width of the blob (or input image) */
	int height;					/**< Height of the blob (or input image) */
	int buffer_type;			/**< Not used reserve parameter, 0 for default (means local memory data)*/
};
typedef struct SeetaCNN_InputOutputData SeetaCNN_InputOutputData;


/**
* @brief The global error code
*/
enum SeetaCNN_ErrorCode
{
	NOERROR = 0,		/**< No error */
	UNIDENTIFIED_LAYER = 1,		/**< Got an unidentified layer */
	MISSMATCH_DEVICE_ID = 2,	/**< Missmatch  */
};
typedef enum SeetaCNN_ErrorCode SeetaCNN_ErrorCode;

#endif