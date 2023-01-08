#pragma once

/**
 * @file concurrent_ring_buffer.tcc
 * @author Marcel Flottmann
 * @author Julian Gaal
 * @author Pascal Buschermoeller
 */

template<typename T>
ConcurrentRingBuffer<T>::ConcurrentRingBuffer(size_t capacity)
    : buffer_(capacity),
      size_(0),
      pushIdx_(0),
      popIdx_(0),
      mutex_({}),
      cvEmpty_({}),
      cvFull_({})
{
}

template<typename T>
bool ConcurrentRingBuffer<T>::push_nb(const T& val, bool force)
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (size_ == buffer_.size())
    {
        if (force)
        {
            doPop(nullptr);
        }
        else
        {
            return false;
        }
    }

    doPush(val);
    cvEmpty_.notify_one();
    return true;
}

template<typename T>
void ConcurrentRingBuffer<T>::push(const T& val)
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (full())
    {
        cvFull_.wait(lock, [&] { return size_ < buffer_.size(); });
    }

    doPush(val);
    cvEmpty_.notify_one();
}

template<typename T>
bool ConcurrentRingBuffer<T>::pop_nb(T* val, uint32_t timeout_ms)
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (empty())
    {
        if(timeout_ms == 0 || !cvEmpty_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] { return size_ != 0; }))
        {
            return false;
        }
    }

    doPop(val);
    cvFull_.notify_one();
    return true;
}

template<typename T>
void ConcurrentRingBuffer<T>::pop(T* val)
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (empty())
    {
        cvEmpty_.wait(lock, [&] { return size_ != 0; });
    }

    doPop(val);
    cvFull_.notify_one();
}

template<typename T>
bool ConcurrentRingBuffer<T>::pop_nb_if(T* val, std::function<bool(T&)> func, uint32_t timeout_ms)
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (empty())
    {
        if(timeout_ms == 0 || !cvEmpty_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] { return size_ != 0; }))
        {
            return false;
        }
    }

    if (func(buffer_[popIdx_]))
    {
        doPop(val);
    }
    else //das ist kriminell
    {
        return false;
    }

    cvFull_.notify_one();

    return true;
}


template<typename T>
bool ConcurrentRingBuffer<T>::pop_if(T* val, std::function<bool(T&)> func)
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (empty())
    {
        cvEmpty_.wait(lock, [&] { return size_ != 0; });
    }

    if (func(buffer_[popIdx_]))
    {
        doPop(val);
    }
    else //das ist kriminell
    {
        return false;
    }

    cvFull_.notify_one();

    return true;
}

template<typename T>
bool ConcurrentRingBuffer<T>::peek_nb(T* peeker, uint32_t timeout_ms)
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (empty())
    {
        if(timeout_ms == 0 || !cvEmpty_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] { return size_ != 0; }))
        {
            return false;
        }
    }

    *peeker = buffer_[popIdx_];
    return true;
}

template<typename T>
void ConcurrentRingBuffer<T>::peek(T* val)
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (empty())
    {
       cvEmpty_.wait(lock, [&] { return size_ != 0; });
    }

    *val = buffer_[popIdx_];
}

template<typename T>
void ConcurrentRingBuffer<T>::clear()
{
    std::unique_lock<std::mutex> lock(mutex_);
    std::fill(buffer_.begin(), buffer_.end(), T());
    size_ = 0;
    pushIdx_ = 0;
    popIdx_ = 0;
    cvFull_.notify_all();
}

template<typename T>
void ConcurrentRingBuffer<T>::doPush(const T& val)
{
    buffer_[pushIdx_] = val;
    size_++;
    pushIdx_++;

    if (pushIdx_ == buffer_.size())
    {
        pushIdx_ = 0;
    }
}

template<typename T>
void ConcurrentRingBuffer<T>::doPop(T* val)
{
    if (val != nullptr)
    {
        *val = buffer_[popIdx_];
    }

    size_--;
    popIdx_++;

    if (popIdx_ == buffer_.size())
    {
        popIdx_ = 0;
    }
}