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
auto normalize(T* buf, dims_type dims, void** meta) -> int;
template <typename T>
auto inv_normalize(T* buf, dims_type dims, const void* meta) -> int;
auto retrieve_norm_meta_len(const void* meta) -> size_t;  // In number of bytes


template <typename T>
auto normalize2(T* buf, dims_type dims, void** meta) -> int;
template <typename T>
auto inv_normalize2(T* buf, dims_type dims, const void* meta) -> int;


//
// Helper functions that are not supposed to be used by end users.
//
auto calc_log_meta_len(size_t buf_len, uint8_t treatment) -> size_t;  // In number of bytes
auto pack_8_booleans(std::array<bool, 8>) -> uint8_t;
auto unpack_8_booleans(uint8_t) -> std::array<bool, 8>;

template <typename T>
void extract_YZ_slice(const T* src, dims_type dims, size_t x, T* dst); 
template <typename T>
void restore_YZ_slice(T* dst, dims_type dims, size_t x, const T* src); 
template <typename T>
auto calc_RMS(const T* buf, size_t len) -> T;

}; // end of namespace

#endif
