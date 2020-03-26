//
// Created by lby on 2018/3/11.
//

#ifndef _ORZ_EXCEPTION_H
#define _ORZ_EXCEPTION_H

#include <exception>

#include <functional>
#include <sstream>
#include <cmath>
#include <string>
#include <iostream>

#include <iomanip>

namespace holiday {
    namespace orz {
        class OutOfMemoryException : public std::logic_error {
        public:
            explicit OutOfMemoryException(size_t failed_size, const std::string &device, int id = 0)
                : logic_error(OutOfMemoryMessage(device, failed_size, id)), m_device(device), m_failed_size(failed_size), m_id(id){
            }

            static std::string OutOfMemoryMessage(const std::string &device, size_t failed_size, int id) {
                std::ostringstream oss;
                oss << "[ERROR] " << "No enough memory on " << device << ":" << id
                    << ", " << failed_size << "B needed.";
                auto msg = oss.str();
                oss << std::endl;
                std::cerr << oss.str();
                return msg;
            }

            const std::string &device() const {
                return m_device;
            }

            int id() const {
                return m_id;
            }

            size_t failed_size() const {
                return m_failed_size;
            }

        private:
            std::string m_device;
            size_t m_failed_size;
            int m_id;
        };
    }
}
using namespace holiday;

#endif //_ORZ_EXCEPTION_H
