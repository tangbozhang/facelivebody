#ifndef _SEETA_MODEL_HEADER_H
#define _SEETA_MODEL_HEADER_H

#include <sstream>
#include <cstdint>
#include <string>
#include <cstring>

namespace seeta
{
	class ModelHeader
	{
	public:
		int32_t feature_size;
		int32_t channels;
		int32_t width;
		int32_t height;
		std::string blob_name;

		/**
		 * \brief 
		 * \param buffer The buffer reading header
		 * \return The size read
		 */
		size_t read(const char *buffer, size_t size)
		{
			std::istringstream iss(std::string(buffer, size), std::ios::binary);

			iss.read(reinterpret_cast<char *>(&feature_size), sizeof(int32_t));
			iss.read(reinterpret_cast<char *>(&channels), sizeof(int32_t));
			iss.read(reinterpret_cast<char *>(&width), sizeof(int32_t));
			iss.read(reinterpret_cast<char *>(&height), sizeof(int32_t));
			int32_t blob_name_size;
			iss.read(reinterpret_cast<char *>(&blob_name_size), sizeof(int32_t));
			char blob_name_tmp[128] = { 0 };
			iss.read(blob_name_tmp, blob_name_size);
			blob_name = blob_name_tmp;
			return sizeof(int32_t) * 5 + blob_name_size;
		}

		/**
		 * \brief 
		 * \param buffer The buffer writing head
		 * \return The size wrote
		 */
		size_t write(char *buffer, size_t size) const
		{
			std::ostringstream oss(std::ios::binary);
			oss.write(reinterpret_cast<const char *>(&feature_size), sizeof(int32_t));
			oss.write(reinterpret_cast<const char *>(&channels), sizeof(int32_t));
			oss.write(reinterpret_cast<const char *>(&width), sizeof(int32_t));
			oss.write(reinterpret_cast<const char *>(&height), sizeof(int32_t));
			int32_t blob_name_size = static_cast<int32_t>(blob_name.length());
			oss.write(reinterpret_cast<const char *>(&blob_name_size), sizeof(int32_t));
			oss.write(blob_name.c_str(), blob_name_size);
			std::string wrote_buffer = oss.str();
			size_t wrote_size = wrote_buffer.size();
#if _MSC_VER >= 1600			
			memcpy_s(buffer, size, wrote_buffer.data(), wrote_size);
#else // _MSC_VER
            memcpy(buffer, wrote_buffer.data(), wrote_size);
#endif // _MSC_VER            

			return wrote_size;
		}
	};
}

#endif
