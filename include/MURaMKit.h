#ifndef MKIT_H
#define MKIT_H

#include <stddef.h>  /* size_t */
#include <stdint.h>  /* fixed width integers */

#ifdef __cplusplus
namespace mkit {
extern "C" {
#endif

int smart_log(
    void*   buf,        /* Input and Output: buffer of double values */
    int     type,       /* Input: buf data type: 1 == float, 0 == double */
    size_t  buf_len,    /* Input: number of double values */
    void**  meta);      /* Output: meta data that records necessary info for exp operations */
                        /* Note: meta points to another pointer which points to NULL. */

int smart_exp(
    void*   buf,        /* Input and Output: buffer of double values */
    int     type,       /* Input: buf data type: 1 == float, 0 == double */
    size_t  buf_len,    /* Input: number of double values */
    void*   meta);      /* Input: meta data that is needed for exp operations */

/* size_t read_meta_len(void* meta); */

#ifdef __cplusplus
} /* end of extern "C" */

template <typename T>
int smart_log_impl(T* buf, size_t buf_len, void** meta);

} /* end of namespace */
#endif

#endif
