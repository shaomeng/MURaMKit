#ifndef MURAMKIT_CAPI
#define MURAMKIT_CAPI

#include <stddef.h> /* for size_t */
#include <stdint.h> /* for fixed-width integers */

#ifdef __cplusplus
namespace C_API {
extern "C" {
#endif

int mkit_smart_log(
    void*   buf,      /* Input and Output: a buffer of double or float values */
    int     is_float, /* Input: data type: 1 == float, 0 == double */
    size_t  buf_len,  /* Input: number of values in buf */
    void**  meta);    /* Output: the meta data needed to perform a smart exp */

int mkit_smart_exp(
    void*       buf,      /* Input and Output: a buffer of double or float values */
    int         is_float, /* Input: data type: 1 == float, 0 == double */
    size_t      buf_len,  /* Input: number of values in buf */
    const void* meta);    /* Input: meta data needed to perform the exp operation */

size_t mkit_log_meta_len(
    const void* meta);    /* Input: meta data from a smart log operation */

#ifdef __cplusplus
}  /* end of extern "C" */
}; /* end of namespace C_API */
#endif


#endif
