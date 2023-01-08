#pragma once
#include <mutex>
#include <vector>
#include <condition_variable>
#include <functional>

constexpr uint32_t DEFAULT_POP_TIMEOUT = 100;

/**
 * @brief Ring buffer with multithreading support.
 *
 * @tparam T Type of the buffer.
 */
template<typename T>
class ConcurrentRingBuffer
{
public:
    /**
     * @brief Construct a new Concurrent Ring Buffer object with specified size.
     *
     * @param size Size of the buffer.
     */
    explicit ConcurrentRingBuffer(size_t size);

    /**
     * @brief Push element non-blocking to the ring buffer.
     *
     * @param val Element to push.
     * @param force When true, the oldest element will be deleted if the buffer is full.
     * @return true Element was sucessfully pushed.
     * @return false The buffer is full.
     */
    bool push_nb(const T& val, bool force = false);

    /**
     * @brief Push element to the ring bufer. Blocks until an element is popped or the the buffer is cleared.
     *
     * @param val Element to push.
     */
    void push(const T& val);

    /**
     * @brief Pop element from the ring buffer
     *
     * @param val Pointer to element to fill with popped value. If val is a nullptr no value is assigned, but an element is popped nonetheless.
     * @param timeout_ms Time to wait for element to pop.
     * @return true Element popped successfully.
     * @return false The ring buffer is empty.
     */
    bool pop_nb(T* val, uint32_t timeout_ms = 0);

    /**
     * @brief Pop element from the ring buffer. Blocks until an element is pushed.
     *
     * @param val Pointer to element to fill with popped value. If val is a nullptr no value is assigned, but an element is popped nonetheless.
     */
    void pop(T* val);

    /**
     * Pop non blocking if function evaluates to true
     * @param val object to pop
     * @param func function to evaluate
     * @param timeout_ms timeout to wait before returning, if no data present
     * @return true if popped
     * @return false if !popped: timeout or function didn't evaluate to true
     */
    bool pop_nb_if(T* val, std::function<bool(T&)> func, uint32_t timeout_ms = 0);

    /**
     * Pop (wait until data available) if function evaluates to true
     * @param val object to pop
     * @param func Function to evaluate
     * @return true if popped
     * @return false if function didn't evaluate to true
     */
    bool pop_if(T* val, std::function<bool(T&)> func);


    /**
     * Peek a value (non blocking): aka don't pop from buffer, but provide a copy to look at
     * @param val saves copied object into val
     * @param timeout_ms timeout to wait for data
     * @return true if successfully peeked
     * @return false timeout passed
     */
    bool peek_nb(T* val, uint32_t timeout_ms = 0);

    /**
     * Peek a value: aka don't pop from buffer, but provide a copy to look at
     * @param val value to save copy in
     */
    void peek(T* val);

    /**
     * @brief Clear the buffer
     *
     */
    void clear();

    /**
     * @brief return current buffer size
     * @return buffer size
     */
    inline size_t size() const
    {
        return size_;
    }

    /**
     * @brief return total capacity of concurrent ring buffer
     * @return buffer capacity
     */
    inline size_t capacity() const
    {
        return buffer_.capacity();
    }

    /**
     * @brief Check if buffer is empty
     * 
     * @return true if empty
     * @return false if not empty
     */
    bool empty() const
    {
        return size_ == 0;
    }

    /**
     * @brief Check if buffer is full
     * 
     * @return true if full
     * @return false if not full
     */
    bool full() const
    {
        return size_ == buffer_.size();
    }

    using Ptr = std::shared_ptr<ConcurrentRingBuffer<T>>;

private:
    /**
     * @brief Actually do the push
     *
     * @param val Element to push.
     */
    void doPush(const T& val);

    /**
     * @brief Actually do the pop
     *
     * @param val Pointer to element to fill popped value. If val is a nullptr no value is assigned, but an element is popped nonetheless.
     */
    void doPop(T* val);

    /// Buffer containing the data
    std::vector<T> buffer_;

    /// Current size of the bufffer
    size_t size_;

    /// Current push position
    size_t pushIdx_;

    /// Current pop position
    size_t popIdx_;

    /// Mutex for locking
    std::mutex mutex_;

    /// Condition variable to wait on, if the ring buffer is empty
    std::condition_variable cvEmpty_;

    /// Condition variable to wait on, if the ring bufer is full
    std::condition_variable cvFull_;
};

#include "concurrent_ring_buffer.tcc"
