#pragma once

#include "VIPLStruct.h"
#include <mutex>

class BaseAction
{
public:
	virtual ~BaseAction() {}

	/**
	 * \brief reset state, so can do next detection
	 */
	virtual void reset() = 0;

	/**
	 * \brief detect this action if appear
	 * \param img 
	 * \param info 
	 * \param points5 
	 * \return 
	 */
	virtual bool detect(const VIPLImageData &img, const VIPLFaceInfo &info, const VIPLPoint *points5) = 0;
};

template <typename T>
class UpdateValue
{
public:
    using Result = T;

    virtual ~UpdateValue() {}
    virtual T update() const = 0;
};

template <typename T>
class CommonValue : public UpdateValue<T>
{
public:
    CommonValue() : m_value(), m_updated(false) {}
    CommonValue(const T &value) : m_value(value), m_updated(true) {}

    T get() const
    {
        std::unique_lock<std::mutex> _locker(m_mutex);

        if (!this->m_updated)
        {
            this->m_value = this->update();
            this->m_updated = true;
        }
        return this->m_value;
    }

    void set(const T &value)
    {
        std::unique_lock<std::mutex> _locker(m_mutex);
        this->m_value = value;
        this->m_updated = true;
    }

    void clear()
    {
        std::unique_lock<std::mutex> _locker(m_mutex);
        this->m_updated = false;
    }

    operator T() const
    {
        return this->get();
    }

private:
    mutable T m_value;
    mutable bool m_updated = false;
    mutable std::mutex m_mutex;
};