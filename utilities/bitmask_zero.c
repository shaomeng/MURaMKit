#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "MURaMKit_CAPI.h"

#define FLT double

int main(int argc, char* argv[])
{
  char* infile = NULL;

  if (argc == 2) {
    infile = argv[1];
  }
  else {
    printf("Usage: ./bitmask  input_file \n");
    return __LINE__;
  }

  /* read infile */
  FILE* f = fopen(infile, "r");
  if (!f) {
    printf("!! input file doesn't exist: %s\n", infile);
    return __LINE__;
  }
  fseek(f, 0, SEEK_END);
  const long len = ftell(f) / sizeof(FLT);  // number of floats/doubles
  if (ftell(f) % sizeof(FLT) != 0) {
    printf("!! input file size error!\n");
    return __LINE__;
  }
  else
    printf("-- analysis: input has %ld values!\n", len);

  fseek(f, 0, SEEK_SET);
  FLT* inbuf = (FLT*)malloc(len * sizeof(FLT));
  size_t tmp = fread(inbuf, sizeof(FLT), len, f);
  assert(tmp == len);
  fclose(f);

  /* Compress using mkit_bitmask_zero() */
  void* comp = NULL;
  int rtn = mkit_bitmask_zero(inbuf, sizeof(FLT) == 4, len, &comp);
  if (rtn) {
    printf("Compression failed!\n");
    return __LINE__;
  }
  else
    printf("-- analysis: compression ratio = %.2fX\n", 
           (double)len * (double)sizeof(FLT) / (double)mkit_bitmask_zero_buf_len(comp));

  /* Decompress using mkit_inv_bitmask_zero() */
  void* output = NULL;
  rtn = mkit_inv_bitmask_zero(comp, &output);
  if (rtn) {
    printf("Decompression failed!\n");
    return __LINE__;
  }

  /* Compare input and output */
  FLT max = 0.0;
  if (sizeof(FLT) == 4) {
    const float* p1 = (const float*)inbuf;
    const float* p2 = (const float*)output;
    for (size_t i = 0; i < len; i++)
      if (fabsf(p1[i] - p2[i]) > max)
        max = fabsf(p1[i] - p2[i]);
  }
  else {
    const double* p1 = (const double*)inbuf;
    const double* p2 = (const double*)output;
    for (size_t i = 0; i < len; i++)
      if (fabs(p1[i] - p2[i]) > max)
        max = fabs(p1[i] - p2[i]);
  }
  printf("-- analysis: compression max diff = %.2e\n", max);

  free(inbuf);  
  free(comp);  
  free(output);  
}
