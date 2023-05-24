#include "MURaMKit.h"
#include "Bitmask.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

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
  uint8_t* tmp_buf = new uint8_t[meta_len];
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
        mask.write_bit(i, false);
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
        mask.write_bit(i, true);
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
