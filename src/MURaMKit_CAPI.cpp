#include "MURaMKit_CAPI.h"

#include "MURaMKit.h"

int C_API::mkit_smart_log(void* buf, int is_float, size_t buf_len, void** meta)
{
  switch (is_float) {
    case 0: {
      double* bufd = static_cast<double*>(buf);
      return mkit::smart_log(bufd, buf_len, meta);
    }
    case 1: {
      float* buff = static_cast<float*>(buf);
      return mkit::smart_log(buff, buf_len, meta);
    }
    default:
      return -1;
  }
}

int C_API::mkit_smart_exp(void* buf, int is_float, size_t buf_len, const void* meta)
{
  switch (is_float) {
    case 0: {
      double* bufd = static_cast<double*>(buf);
      return mkit::smart_exp(bufd, buf_len, meta);
    }
    case 1: {
      float* buff = static_cast<float*>(buf);
      return mkit::smart_exp(buff, buf_len, meta);
    }
    default:
      return -1;
  }
}

size_t C_API::mkit_log_meta_len(const void* meta)
{
  return mkit::retrieve_log_meta_len(meta);
}

int C_API::mkit_slice_norm(void* buf,
                           int is_float,
                           size_t dim_fast,
                           size_t dim_mid,
                           size_t dim_slow,
                           void** meta)
{
  const auto dims = mkit::dims_type{dim_fast, dim_mid, dim_slow};
  switch (is_float) {
    case 0: {
      double* bufd = static_cast<double*>(buf);
      return mkit::slice_norm(bufd, dims, meta);
    }
    case 1: {
      float* buff = static_cast<float*>(buf);
      return mkit::slice_norm(buff, dims, meta);
    }
    default:
      return -1;
  }
}

int C_API::mkit_inv_slice_norm(void* buf,
                               int is_float,
                               size_t dim_fast,
                               size_t dim_mid,
                               size_t dim_slow,
                               const void* meta)
{
  const auto dims = mkit::dims_type{dim_fast, dim_mid, dim_slow};
  switch (is_float) {
    case 0: {
      double* bufd = static_cast<double*>(buf);
      return mkit::inv_slice_norm(bufd, dims, meta);
    }
    case 1: {
      float* buff = static_cast<float*>(buf);
      return mkit::inv_slice_norm(buff, dims, meta);
    }
    default:
      return -1;
  }
}

size_t C_API::mkit_slice_norm_meta_len(const void* meta)
{
  return mkit::retrieve_slice_norm_meta_len(meta);
}

int C_API::mkit_bitmask_zero(const void* inbuf,
                             int is_float,
                             size_t len,
                             void** output)
{
  switch (is_float) {
    case 0: {
      const double* bufd = static_cast<const double*>(inbuf);
      return mkit::bitmask_zero(bufd, len, output);
    }
    case 1: {
      const float* buff = static_cast<const float*>(inbuf);
      return mkit::bitmask_zero(buff, len, output);
    }
    default:
      return -1;
  }
}

int C_API::mkit_inv_bitmask_zero(const void* inbuf,
                                 void** output)
{
  return mkit::inv_bitmask_zero(inbuf, output);
}

size_t C_API::mkit_bitmask_zero_buf_len(const void* inbuf)
{
  return mkit::retrieve_bitmask_zero_buf_len(inbuf);
}
