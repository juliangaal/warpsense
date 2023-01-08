/**
 * @file concurrent_ringbuffer.cpp
 * @author Juri Vana
 * @date 2020-10-13
 */

#include <gtest/gtest.h>
#include <util/concurrent_ring_buffer.h>
#include <iostream>
#include <thread>

TEST(warpsense, crb_push_pop)
{
    // init
    constexpr size_t buffer_size = 5;
    ConcurrentRingBuffer<size_t> crb(buffer_size);

    EXPECT_EQ(crb.size(), 0);
    EXPECT_EQ(crb.capacity(), buffer_size);
    EXPECT_TRUE(crb.empty());

    for (size_t i = 0; i < buffer_size; ++i)
    {
        crb.push(i);
    }

    EXPECT_EQ(crb.size(), buffer_size);
    EXPECT_TRUE(crb.full());

    // test push_nb, pop
    EXPECT_TRUE(!crb.push_nb(0));
    EXPECT_TRUE(crb.push_nb(buffer_size, true));

    // Pop first value
    size_t val;
    crb.pop(&val);
    EXPECT_EQ(val, 1);
    EXPECT_EQ(crb.size(), buffer_size - 1);

    // test pop_nb, buffer begins with 2
    for (size_t i = 2; i <= buffer_size; i++)
    {
        EXPECT_TRUE(crb.pop_nb(&val));
        EXPECT_EQ(val, i);
        EXPECT_EQ(crb.size(), buffer_size - i);
    }

    EXPECT_EQ(crb.size(), 0);
    EXPECT_TRUE(!crb.pop_nb(&val));
}

TEST(warpsense, crb_clear)
{
    // init
    constexpr size_t buffer_size = 5;
    ConcurrentRingBuffer<size_t> crb(buffer_size);

    EXPECT_EQ(crb.size(), 0);
    EXPECT_EQ(crb.capacity(), buffer_size);
    EXPECT_TRUE(crb.empty());

    for (size_t i = 0; i < buffer_size; ++i)
    {
        crb.push(i);
    }

    EXPECT_EQ(crb.size(), buffer_size);
    EXPECT_TRUE(crb.full());

    // test clear
    crb.clear();
    EXPECT_EQ(crb.size(), 0);
    EXPECT_TRUE(crb.empty());
}

TEST(warpsense, crb_multithreading)
{
    // init
    constexpr size_t buffer_size = 5;
    ConcurrentRingBuffer<size_t> crb(buffer_size);

    EXPECT_EQ(crb.size(), 0);
    EXPECT_EQ(crb.capacity(), buffer_size);
    EXPECT_TRUE(crb.empty());

    for (size_t i = 0; i < buffer_size; ++i)
    {
        crb.push(i);
    }

    EXPECT_EQ(crb.size(), buffer_size);
    EXPECT_TRUE(crb.full());

    size_t val;
    size_t length;

    EXPECT_EQ(crb.size(), buffer_size);

    // test waiting to push
    std::thread push_thread{[&]()
    {
        // wait for other thread to pop
        crb.push(buffer_size);
        length = crb.size();
    }};

    std::thread pop_thread{[&]()
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        crb.pop(&val);
    }};

    push_thread.join();
    pop_thread.join();

    EXPECT_EQ(length, buffer_size);
    EXPECT_EQ(val, 0);

    // test waiting to pop
    crb.clear();

    std::thread push_thread2{[&]()
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        crb.push(1);
    }};

    std::thread pop_thread2{[&]()
    {
        // wait for other thread to push
        crb.pop(&val);
        length = crb.size();
    }};

    push_thread2.join();
    pop_thread2.join();

    EXPECT_EQ(length, 0);
    EXPECT_EQ(val, 1);

    // after all these changes, capcity should not change
    EXPECT_EQ(crb.capacity(), buffer_size);
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}