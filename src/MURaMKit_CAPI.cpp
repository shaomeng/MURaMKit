#include "MURaMKit_CAPI.h"

#include "MURaMKit.h"


int C_API::mkit_smart_log(void* buf, int is_float, size_t buf_len, void** meta)
{
  switch (is_float) {
    case 0:
    {
      double* bufd = static_cast<double*>(buf);
      return mkit::smart_log(bufd, buf_len, meta);
    }
    case 1:
    {
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
    case 0:
    {
      double* bufd = static_cast<double*>(buf);
      return mkit::smart_exp(bufd, buf_len, meta);
    }
    case 1:
    {
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

