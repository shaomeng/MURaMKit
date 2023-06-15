#include "MURaMKit.h"
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
  auto has_neg = std::any_of(buf, buf + buf_len, [](auto v){ return v < 0.0; });
  auto has_zero = std::any_of(buf, buf + buf_len, [](auto v){ return v == 0.0; });

  // Step 2: record test results
  auto treatment = pack_8_booleans({has_neg, has_zero, false, false, false, false, false, false});

  // Step 3: calculate meta field total size, and fill in `buf_len` and `treatment`
  auto meta_len = calc_log_meta_len(buf_len, treatment);
  uint8_t* tmp_buf = static_cast<uint8_t*>(std::malloc(meta_len));
  auto tmp64 = uint64_t{buf_len};
  std::memcpy(tmp_buf, &tmp64, sizeof(tmp64));
  tmp_buf[8] = treatment;
  size_t pos = 9;

  // Step 4: apply conditioning operations
  //
  auto mask = Bitmask();

  // Step 4.1: make all values non-negative
  if (has_neg) {
    mask.resize(buf_len);
    mask.reset_true();
    for (size_t i = 0; i < buf_len; i++) {
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
    for (size_t i = 0; i < buf_len; i++) {
      if (buf[i] == 0.0)
        mask.write_true(i);
      else
        buf[i] = std::log(buf[i]);
    }
    const auto& mask_buf = mask.view_buffer();
    auto mask_num_bytes = mask_buf.size() * 8;
    std::memcpy(tmp_buf + pos, mask_buf.data(), mask_num_bytes);
  }
  else
    std::for_each(buf, buf + buf_len, [](auto& v){ v = std::log(v); });

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
  std::for_each(buf, buf + buf_len, [](auto& v){ v = std::exp(v); });
  auto mask = Bitmask();
  if (has_zero) {
    // Need to figure out where zero mask is stored
    mask.resize(buf_len);
    size_t pos = 9;
    if (has_neg) 
      pos += mask.view_buffer().size() * 8;
    mask.use_bitstream(p + pos);
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
    for (size_t i = 0; i < buf_len; i++)
      if (!mask.read_bit(i))
        buf[i] = -buf[i];
  }

  return 0;
}
template auto mkit::smart_exp(float* buf, size_t buf_len, const void* meta) -> int;
template auto mkit::smart_exp(double* buf, size_t buf_len, const void* meta) -> int;

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
auto mkit::normalize(T* buf, dims_type dims, void** meta) -> int
{
  if (*meta != nullptr)
    return 1;

  // In case of 2D slices, really does nothing, just record a header size of 4 bytes.
  if (dims[2] == 1) {
    uint32_t len = sizeof(uint32_t);
    uint8_t* tmp_buf = (uint8_t*)std::malloc(len);
    std::memcpy(tmp_buf, &len, sizeof(len));
    *meta = tmp_buf;
    return 0;
  }

  //
  // Continue with 3D cases.
  //
  const auto slice_len = dims[1] * dims[2];
  auto slice_buf = std::make_unique<T[]>(slice_len);

  // Filter header definition:
  // Length (uint32_t) +  pairs of mean and rms (double + double)
  //
  const uint32_t len = sizeof(uint32_t) + sizeof(double) * 2 * dims[0];
  uint8_t* tmp_buf = (uint8_t*)std::malloc(len);
  std::memcpy(tmp_buf, &len, sizeof(len));
  double* const body = reinterpret_cast<double*>(tmp_buf + sizeof(len));

  for (size_t x = 0; x < dims[0]; x++) {
    extract_YZ_slice(buf, dims, x, slice_buf.get());

    // Operation 1: subtract mean
    auto mean = std::accumulate(slice_buf.get(), slice_buf.get() + slice_len, T{0.0});
    mean /= T(slice_len);
    std::for_each(slice_buf.get(), slice_buf.get() + slice_len, [mean](auto& v) { v -= mean; });

    // Operation 2: divide by RMS
    auto rms = calc_RMS(slice_buf.get(), slice_len);
    if (rms == 0.0)
      rms = 1.0;
    std::for_each(slice_buf.get(), slice_buf.get() + slice_len, [rms](auto& v) { v /= rms; });

    restore_YZ_slice(buf, dims, x, slice_buf.get());

    // Save mean and rms data
    body[x * 2] = mean;
    body[x * 2 + 1] = rms;
  }

  *meta = tmp_buf;
  return 0;
}
template auto mkit::normalize(float* buf, dims_type dims, void** meta) -> int;
template auto mkit::normalize(double* buf, dims_type dims, void** meta) -> int;

template <typename T>
auto mkit::inv_normalize(T* buf, dims_type dims, const void* meta) -> int
{
  // In case of 2D slices, really does nothing.
  if (dims[2] == 1)
    return true;

  //
  // Continue with 3D cases.
  //
  const auto slice_len = dims[1] * dims[2];
  auto slice_buf = std::make_unique<T[]>(slice_len);
  const double* const body =
      reinterpret_cast<const double*>(static_cast<const uint8_t*>(meta) + 4);

  for (size_t x = 0; x < dims[0]; x++) {
    const auto mean = body[x * 2];
    const auto rms = body[x * 2 + 1];
    extract_YZ_slice(buf, dims, x, slice_buf.get());

    // Operation 1: multiply by RMS
    std::for_each(slice_buf.get(), slice_buf.get() + slice_len, [rms](auto& v) { v *= rms; });

    // Operation 2: add mean
    std::for_each(slice_buf.get(), slice_buf.get() + slice_len, [mean](auto& v) { v += mean; });

    restore_YZ_slice(buf, dims, x, slice_buf.get());
  }

  return 0;
}
template auto mkit::inv_normalize(float* buf, dims_type dims, const void* meta) -> int;
template auto mkit::inv_normalize(double* buf, dims_type dims, const void* meta) -> int;

auto mkit::retrieve_norm_meta_len(const void* meta) -> size_t
{
  // Directly read the first 4 bytes
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

template <typename T>
void mkit::extract_YZ_slice(const T* src, dims_type dims, size_t x, T* dst)
{
  assert(x < dims[0]);

  const auto YZ_slice = dims[1] * dims[2];

  size_t idx = 0;
  const auto XY_plane = dims[0] * dims[1];
  for (size_t z = 0; z < dims[2]; z++)
    for (size_t y = 0; y < dims[1]; y++)
      dst[idx++] = src[z * XY_plane + y * dims[0] + x];
}
template void mkit::extract_YZ_slice(const float* src, dims_type dims, size_t x, float* dst);
template void mkit::extract_YZ_slice(const double* src, dims_type dims, size_t x, double* dst);

template <typename T>
void mkit::restore_YZ_slice(T* dst, dims_type dims, size_t x, const T* src)
{
  assert(x < dims[0]);
  const auto YZ_slice = dims[1] * dims[2];

  size_t idx = 0;
  const auto XY_plane = dims[0] * dims[1];
  for (size_t z = 0; z < dims[2]; z++)
    for (size_t y = 0; y < dims[1]; y++)
      dst[z * XY_plane + y * dims[0] + x] = src[idx++];
}
template void mkit::restore_YZ_slice(float* dst, dims_type dims, size_t x, const float* src);
template void mkit::restore_YZ_slice(double* dst, dims_type dims, size_t x, const double* src);

template <typename T>
auto mkit::calc_RMS(const T* buf, size_t len) -> T
{
  auto sum = std::accumulate(buf, buf + len, T{0.0}, [](auto a, auto b) { return a + b * b; });
  sum /= T(len);
  sum = std::sqrt(sum);
  return sum;
}
template auto mkit::calc_RMS(const float* buf, size_t len) -> float;
template auto mkit::calc_RMS(const double* buf, size_t len) -> double;
