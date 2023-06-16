#ifndef MKIT_H
#define MKIT_H

#include <array>
#include <cstddef>  // size_t
#include <cstdint>  // fixed width integers
#include <vector>  // fixed width integers

namespace mkit {

using std::size_t;
using dims_type = std::array<size_t, 3>;

template <typename T>
auto smart_log(T* buf, size_t buf_len, void** meta) -> int;
template <typename T>
auto smart_exp(T* buf, size_t buf_len, const void* meta) -> int;
auto retrieve_log_meta_len(const void* meta) -> size_t;  // In number of bytes


template <typename T>
auto slice_norm(T* buf, dims_type dims, void** meta) -> int;
template <typename T>
auto inv_slice_norm(T* buf, dims_type dims, const void* meta) -> int;
auto retrieve_slice_norm_meta_len(const void* meta) -> size_t;  // In number of bytes


//
// Helper functions that are not supposed to be used by end users.
//
auto calc_log_meta_len(size_t buf_len, uint8_t treatment) -> size_t;  // In number of bytes
auto pack_8_booleans(std::array<bool, 8>) -> uint8_t;
auto unpack_8_booleans(uint8_t) -> std::array<bool, 8>;

}; // end of namespace

#endif
