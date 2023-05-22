#include "MURaMKit.h"
#include "Bitmask.h"
#include "zstd.h"

#include <algorithm>
#include <cmath>
#include <numeric>

int mkit::smart_log(void*   buf,
                    int     type,
                    size_t  buf_len,
                    void**  meta)
{
  if (*meta != NULL)
    return 1;  

  return 0;
}

int mkit::smart_exp(void*   buf,
                    int     type,
                    size_t  buf_len,
                    void*   meta)
{
  return 0;
}

template <typename T>
int mkit::smart_log_impl(T* buf, size_t buf_len, void** meta)
{
  auto neg_mask = Bitmask();
  auto zero_mask = Bitmask();

  // Step 1: are there negative values in `buf`?
  auto has_neg = std::any_of(buf, buf + buf_len, [](auto v){ return v < 0.0; });
  if (has_neg) {
    neg_mask.resize(buf_len);
    for (size_t i = 0; i < buf_len; i++) {
      neg_mask.write_bit(i, buf[i] >= 0.0);
      buf[i] = std::abs(buf[i]);
    }
  }

  // Step 2: are there absolute zeros in `buf`?
  auto has_zero = std::any_of(buf, buf + buf_len, [](auto v){ return v == 0.0; });
  if (has_zero) {
    zero_mask.resize(buf_len);
    zero_mask.reset();
    for (size_t i = 0; i < buf_len; i++)
      if (buf[i] == 0.0)
        zero_mask.write_bit(i, true);
  }

  // Step 3: record test results
  uint8_t result = uint8_t{has_neg} + uint8_t{has_zero} * uint8_t{2};

  return 0;
}
