#include "MURaMKit.h"
#include <omp.h>
#include "Bitmask.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>

template <typename T>
auto mkit::smart_log(T* buf, size_t buf_len, void** meta) -> int
{
  if (*meta != nullptr)
    return 1;

  // Step 1: are there negative values and/or absolute zeros in `buf`?
  auto has_neg = false, has_zero = false;

#pragma omp parallel
  {
    if (omp_get_thread_num() == 0)  // The 1st thread
      has_neg = std::any_of(buf, buf + buf_len, [](auto v) { return v < 0.0; });
    if (omp_get_thread_num() == omp_get_max_threads() - 1)  // The last thread
      has_zero = std::any_of(buf, buf + buf_len, [](auto v) { return v == 0.0; });
  }

  // Step 2: record test results
  auto treatment = pack_8_booleans({has_neg, has_zero, false, false, false, false, false, false});

  // Step 3: calculate meta field total size, and fill in `buf_len` and `treatment`.
  auto meta_len = calc_log_meta_len(buf_len, treatment);
  uint8_t* tmp_buf = static_cast<uint8_t*>(std::malloc(meta_len));
  auto tmp64 = uint64_t{buf_len};
  std::memcpy(tmp_buf, &tmp64, sizeof(tmp64));
  tmp_buf[8] = treatment;
  size_t pos = 9;

  // Step 4: apply conditioning operations
  //
  auto mask = Bitmask();        // will be re-used
  const size_t stride = 16384;  // must be a multiplier of 64
  const size_t num_strides = (buf_len - buf_len % stride) / stride;

  // Step 4.1: make all values non-negative
  if (has_neg) {
    mask.resize(buf_len);
    mask.reset_true();

#pragma omp parallel for
    for (size_t s = 0; s < num_strides; s++) {
      for (size_t i = s * stride; i < (s + 1) * stride; i++) {
        if (buf[i] < 0.0) {
          mask.write_false(i);
          buf[i] = std::abs(buf[i]);
        }
      }
    }

    for (size_t i = stride * num_strides; i < buf_len; i++) {
      if (buf[i] < 0.0) {
        mask.write_false(i);
        buf[i] = std::abs(buf[i]);
      }
    }
    const auto& mask_buf = mask.view_buffer();
    auto mask_num_bytes = mask_buf.size() * 8;
    std::memcpy(tmp_buf + pos, mask_buf.data(), mask_num_bytes);
    pos += mask_num_bytes;
  }

  // Step 4.2: apply log operation on non-zero values
  if (has_zero) {
    mask.resize(buf_len);
    mask.reset();

#pragma omp parallel for
    for (size_t s = 0; s < num_strides; s++) {
      for (size_t i = s * stride; i < (s + 1) * stride; i++) {
        if (buf[i] == 0.0)
          mask.write_true(i);
        else
          buf[i] = std::log(buf[i]);
      }
    }

    for (size_t i = stride * num_strides; i < buf_len; i++) {
      if (buf[i] == 0.0)
        mask.write_true(i);
      else
        buf[i] = std::log(buf[i]);
    }
    const auto& mask_buf = mask.view_buffer();
    auto mask_num_bytes = mask_buf.size() * 8;
    std::memcpy(tmp_buf + pos, mask_buf.data(), mask_num_bytes);
  }
  else {

#pragma omp parallel for
    for (size_t s = 0; s < num_strides; s++) {
      for (size_t i = s * stride; i < (s + 1) * stride; i++)
        buf[i] = std::log(buf[i]);
    }

    for (size_t i = stride * num_strides; i < buf_len; i++)
      buf[i] = std::log(buf[i]);
  }

  *meta = tmp_buf;

  return 0;
}
template auto mkit::smart_log(float* buf, size_t buf_len, void** meta) -> int;
template auto mkit::smart_log(double* buf, size_t buf_len, void** meta) -> int;

template <typename T>
auto mkit::smart_exp(T* buf, size_t buf_len, const void* meta) -> int
{
  if (buf_len != static_cast<const uint64_t*>(meta)[0])
    return 1;

  // Step 1: are there negative or absolute zero values?
  //
  const uint8_t* p = static_cast<const uint8_t*>(meta);
  auto [has_neg, has_zero, b2, b3, b4, b5, b6, b7] = unpack_8_booleans(p[8]);

  // Step 2: apply exp to all values, then zero out ones indicated by the zero mask.
  //
#pragma omp parallel for
  for (size_t i = 0; i < buf_len; i++)
    buf[i] = std::exp(buf[i]);

  auto mask = Bitmask();
  if (has_zero) {
    // Need to figure out where zero mask is stored
    mask.resize(buf_len);
    size_t pos = 9;
    if (has_neg)
      pos += mask.view_buffer().size() * 8;
    mask.use_bitstream(p + pos);

#pragma omp parallel for
    for (size_t i = 0; i < buf_len; i++) {
      if (mask.read_bit(i))
        buf[i] = 0.0;
    }
  }

  // Step 3: apply negative signs if needed
  //
  if (has_neg) {
    mask.resize(buf_len);
    mask.use_bitstream(p + 9);

#pragma omp parallel for
    for (size_t i = 0; i < buf_len; i++)
      if (!mask.read_bit(i))
        buf[i] = -buf[i];
  }

  return 0;
}
template auto mkit::smart_exp(float* buf, size_t buf_len, const void* meta) -> int;
template auto mkit::smart_exp(double* buf, size_t buf_len, const void* meta) -> int;

auto mkit::retrieve_log_meta_len(const void* meta) -> size_t
{
  const uint8_t* p = static_cast<const uint8_t*>(meta);

  // Retrieve the first 8 bytes
  auto buf_len = uint64_t{0};
  std::memcpy(&buf_len, p, sizeof(buf_len));

  // Retrieve the 9th byte
  auto treatment = p[8];

  return calc_log_meta_len(buf_len, treatment);
}

template <typename T>
auto mkit::slice_norm(T* buf, dims_type dims, void** meta) -> int
{
  if (*meta != nullptr)
    return 1;

  // In case of 2D slices, really does nothing, just record a header size of 4 bytes.
  //
  if (dims[2] == 1) {
    uint32_t header_len = sizeof(uint32_t);
    void* tmp_buf = std::malloc(header_len);
    std::memcpy(tmp_buf, &header_len, sizeof(header_len));
    *meta = tmp_buf;
    return 0;
  }

  // Filter header definition:
  // Total_length (uint32_t) +  slice means (double) + slice rms (double)
  //
  const auto dimx = dims[0];
  const auto xy = dims[0] * dims[1];
  const auto yz = double(dims[1] * dims[2]);
  const auto total_vals = dims[0] * dims[1] * dims[2];
  const uint32_t header_len = sizeof(uint32_t) + sizeof(double) * 2 * dimx;
  uint8_t* tmp_buf = static_cast<uint8_t*>(std::malloc(header_len));
  std::memcpy(tmp_buf, &header_len, sizeof(header_len));

  // Create a buffer for each OMP thread.
  //
  auto buf_vec = std::vector<std::unique_ptr<double[]>>(omp_get_max_threads());
  for (size_t i = 0; i < buf_vec.size(); i++) {
    buf_vec[i] = std::make_unique<double[]>(dimx);
    std::fill(buf_vec[i].get(), buf_vec[i].get() + dimx, 0.0);
  }

  // First pass: calculate mean
  //
  double* const mean_buf = reinterpret_cast<double*>(tmp_buf + sizeof(header_len));

#pragma omp parallel for
  for (size_t z = 0; z < dims[2]; z++) {
    auto& mybuf = buf_vec[omp_get_thread_num()];
    for (size_t i = z * xy; i < (z + 1) * xy; i++)
      mybuf[i % dimx] += double(buf[i]);
  }

  for (auto& buf : buf_vec) {
    for (size_t i = 0; i < dimx; i++)
      mean_buf[i] += buf[i];
  }
  std::for_each(mean_buf, mean_buf + dimx, [yz](auto& v) { v /= yz; });

// Second pass: subtract mean
//
#pragma omp parallel for
  for (size_t i = 0; i < total_vals; i++) {
    buf[i] -= T(mean_buf[i % dimx]);
  }

  // Third pass: calculate RMS
  //
  double* const rms_buf = mean_buf + dimx;

#pragma omp parallel for
  for (size_t i = 0; i < buf_vec.size(); i++)
    std::fill(buf_vec[i].get(), buf_vec[i].get() + dimx, 0.0);

#pragma omp parallel for
  for (size_t z = 0; z < dims[2]; z++) {
    auto& mybuf = buf_vec[omp_get_thread_num()];
    for (size_t i = z * xy; i < (z + 1) * xy; i++)
      mybuf[i % dimx] += double(buf[i] * buf[i]);
  }

  for (auto& buf : buf_vec) {
    for (size_t i = 0; i < dimx; i++)
      rms_buf[i] += buf[i];
  }
  std::for_each(rms_buf, rms_buf + dimx, [yz](auto& v) {
    v /= yz;
    v = std::sqrt(v);
  });
  std::replace(rms_buf, rms_buf + dimx, 0.0, 1.0);

// Fourth pass: divide by RMS
//
#pragma omp parallel for
  for (size_t i = 0; i < total_vals; i++)
    buf[i] /= T(rms_buf[i % dimx]);

  *meta = tmp_buf;
  return 0;
}
template auto mkit::slice_norm(float* buf, dims_type dims, void** meta) -> int;
template auto mkit::slice_norm(double* buf, dims_type dims, void** meta) -> int;

template <typename T>
auto mkit::inv_slice_norm(T* buf, dims_type dims, const void* meta) -> int
{
  // In case of 2D slices, really does nothing.
  //
  if (dims[2] == 1)
    return 0;

  const auto dimx = dims[0];
  const auto total_vals = dims[0] * dims[1] * dims[2];
  const double* const mean_buf =
      reinterpret_cast<const double*>(static_cast<const uint8_t*>(meta) + 4);
  const double* const rms_buf = mean_buf + dimx;

#pragma omp parallel for
  for (size_t i = 0; i < total_vals; i++) {
    buf[i] *= T(rms_buf[i % dimx]);
    buf[i] += T(mean_buf[i % dimx]);
  }

  return 0;
}
template auto mkit::inv_slice_norm(float* buf, dims_type dims, const void* meta) -> int;
template auto mkit::inv_slice_norm(double* buf, dims_type dims, const void* meta) -> int;

auto mkit::retrieve_slice_norm_meta_len(const void* meta) -> size_t
{
  // Directly read the first 4 bytes
  //
  uint32_t len = 0;
  std::memcpy(&len, meta, sizeof(len));
  return len;
}

auto mkit::pack_8_booleans(std::array<bool, 8> src) -> uint8_t
{
  auto bytes = std::array<uint8_t, 8>();
  std::copy(src.cbegin(), src.cend(), bytes.begin());
  const uint64_t magic = 0x8040201008040201;
  uint64_t t = 0;
  std::memcpy(&t, bytes.data(), 8);
  uint8_t dest = (magic * t) >> 56;
  return dest;
}

auto mkit::unpack_8_booleans(uint8_t src) -> std::array<bool, 8>
{
  const uint64_t magic = 0x8040201008040201;
  const uint64_t mask = 0x8080808080808080;
  uint64_t t = ((magic * src) & mask) >> 7;
  auto bytes = std::array<uint8_t, 8>();
  std::memcpy(bytes.data(), &t, 8);
  auto b8 = std::array<bool, 8>();
  std::copy(bytes.cbegin(), bytes.cend(), b8.begin());
  return b8;
}

auto mkit::calc_log_meta_len(size_t buf_len, uint8_t treatment) -> size_t
{
  auto num_long = buf_len / 64;
  if (buf_len % 64 != 0)
    num_long++;
  auto [has_neg, has_zero, b2, b3, b4, b5, b6, b7] = unpack_8_booleans(treatment);

  auto meta_len = size_t{9};  // The fixed len field + treatment field.
  if (has_neg)
    meta_len += num_long * 8;
  if (has_zero)
    meta_len += num_long * 8;

  return meta_len;
}
