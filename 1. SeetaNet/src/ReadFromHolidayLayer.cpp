#include "ReadFromHolidayLayer.h"
#include "MemoryModel.h"

#include <string>
#include <vector>
#include <fstream>

#include "memory_stream.h"

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>

/*namespace Holiday
{
    */int ReadStringFromBinaryFile(std::string& input_string, std::memoryi_stream& inputfStream)
    {
        int64_t string_size(0);
        inputfStream.read(reinterpret_cast<char*>(&string_size), sizeof(int64_t));
        input_string.resize(string_size);

        inputfStream.read(const_cast<char*>(input_string.data()), sizeof(char)*string_size);
        return 0;
    }

    int ReadStringVectorFromBinaryFile(std::vector<std::string>& vector_string, std::memoryi_stream& inputfStream)
    {
        int64_t blob_length(0);
        inputfStream.read(reinterpret_cast<char*>(&blob_length), sizeof(int64_t));

        vector_string.resize(blob_length);
        for (int k = 0; k < blob_length; k++)
        {
            ReadStringFromBinaryFile(vector_string[k], inputfStream);
        }
        return 0;
    }

    int ReadAllContentFromFile(const char* inputfilename, char** ppbuffer, int64_t& file_length)
    {
        std::ifstream fin(inputfilename, std::ios::binary | std::ios::in);
        if (!fin.is_open())  {

            return -1;
        }
        fin.seekg(0, std::ios::end);
        file_length = fin.tellg();

        *ppbuffer = new char[file_length];
        fin.seekg(0, std::ios::beg);
        fin.read(*ppbuffer, file_length - 1);
        fin.close();


        return 0;
    }

    template<typename T>
    int ReadCommonLayerFromFile(std::memoryi_stream& input_stream, Holiday_LayerParameter& output_param)
    {
        ;
        std::string read_from_string;
        int64_t length = -1;
        input_stream.read((char*)&length, sizeof(int64_t));

        if (!input_stream)
        {
            return -1;
        }

        read_from_string.resize(length);
        input_stream.read(const_cast<char*>(read_from_string.data()), length);

        char* ptm_current_bufer = new char[length];
        memcpy(ptm_current_bufer, read_from_string.data(), length);
        std::memoryi_stream tmp_memstream(ptm_current_bufer, length);

        google::protobuf::io::ZeroCopyInputStream* raw_input = new google::protobuf::io::IstreamInputStream(&tmp_memstream);
        google::protobuf::io::CodedInputStream* coded_input = new google::protobuf::io::CodedInputStream(raw_input);

        coded_input->SetTotalBytesLimit(INT_MAX, 536870912);
        bool return_result_parse = output_param.ParseFromCodedStream(coded_input);
        //output_param.ParseFromString(read_from_string);
        delete coded_input;
        delete raw_input;
        delete[]ptm_current_bufer;

        return 0;
    }

    int HolidayCNNReadModelFromBuffer(const char* buffer, size_t buffer_length, void** model)
    {
        MemoryModel** tmp_model = (MemoryModel**)model;
        *tmp_model = new MemoryModel;
        if (buffer == nullptr)
        {
            return NULL_PTR;
        }
        std::memoryi_stream fin(buffer, buffer_length);
        ReadStringVectorFromBinaryFile((*tmp_model)->vector_blob_names, fin);
        ReadStringVectorFromBinaryFile((*tmp_model)->vector_layer_names, fin);
        int index_layer = 0;
        int return_result = -1;

        while (!fin.eof())
        {
            return_result = -1;
            
            Holiday_LayerParameter   *output_param = new Holiday_LayerParameter;
            int read_return_value = ReadCommonLayerFromFile<float>(fin, *output_param);

            output_param->set_layer_index(index_layer);
            index_layer++;

            if (output_param->type() == 1001)
            {
                delete output_param;
                break;
            }

            if (0 == read_return_value)
            {
                (*tmp_model)->all_layer_params.push_back(output_param);
            }
        }

        return 0;
	}


	int HolidayCNNReleaseModel(void** model)
	{
		MemoryModel** tmp_model = (MemoryModel**)model;
		for (int i = 0; i < (*tmp_model)->all_layer_params.size(); i++)
		{
			delete (*tmp_model)->all_layer_params[i];
		}
		(*tmp_model)->all_layer_params.clear();
		(*tmp_model)->vector_blob_names.clear();
		(*tmp_model)->vector_layer_names.clear();

		delete *tmp_model;

		*tmp_model = nullptr;

		return 0;
	}

	int HolidayCNNModelResetInput(void* model, int width, int height)
	{
		MemoryModel* tmp_model = (MemoryModel*)model;
		tmp_model->m_new_width = width;
		tmp_model->m_new_height = height;
		return 0;
	}

//};