#ifndef MKIT_H
#define MKIT_H

#include <array>
#include <cstddef>  // size_t
#include <cstdint>  // fixed width integers

namespace mkit {

using std::size_t;

auto pack_8_booleans(std::array<bool, 8>) -> uint8_t;
auto unpack_8_booleans(uint8_t) -> std::array<bool, 8>;

template <typename T>
auto smart_log(T* buf, size_t buf_len, void** meta) -> int;
template <typename T>
auto smart_exp(T* buf, size_t buf_len, const void* meta) -> int;

auto calc_log_meta_len(size_t buf_len, uint8_t treatment) -> size_t;  // In number of bytes
auto retrieve_log_meta_len(const void* meta) -> size_t;  // In number of bytes

}; // end of namespace

#endif
