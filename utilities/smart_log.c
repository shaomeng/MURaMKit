#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "MURaMKit_CAPI.h"

#define FLT double

int main(int argc, char** argv)
{
  char* infile = NULL;
  char* outfile = NULL;
  char* outmeta = NULL;

  if (argc == 2)
    infile = argv[1];
  else if (argc == 4) {
    infile = argv[1];
    outfile = argv[2];
    outmeta = argv[3];
  }
  else {
    printf("Usage: ./smart_log  input_file  [output_file]  [output_metadata]\n");
    return __LINE__;
  }

  /* read infile */
  FILE* f = fopen(infile, "r");
  if (!f) {
    printf("!! input file doesn't exist: %s\n", infile);
    return __LINE__;
  }
  fseek(f, 0, SEEK_END);
  const long len = ftell(f) / sizeof(FLT);  /* number of floats/doubles */
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

  /* if there are negative values or absolute zeros */
  int has_neg = 0, has_zero = 0;
  for (size_t i = 0; i < len; i++)
    if (inbuf[i] < 0.0) {
      has_neg = 1;
      break;
    }
  for (size_t i = 0; i < len; i++)
    if (inbuf[i] == 0.0) {
      has_zero = 1;
      break;
    }
  printf("-- analysis: input has negative values: %d, has absolute zeros: %d\n", has_neg, has_zero);

  /* apply smart log to a copy */
  FLT* outbuf = (FLT*)malloc(len * sizeof(FLT));
  memcpy(outbuf, inbuf, len * sizeof(FLT));
  void* meta = NULL;
  int rtn = mkit_smart_log(outbuf, sizeof(FLT) == 4, len, &meta);
  if (rtn) {
    printf("!! error when applying smart log!\n");
    return __LINE__;
  }
  else
    printf("-- status: successfully applying smart log, meta size = %lu\n", mkit_log_meta_len(meta));

  /* verification: apply smart exp, and print out the maximum difference */
  rtn = mkit_smart_exp(outbuf, sizeof(FLT) == 4, len, meta);
  if (rtn) {
    printf("!! error when applying smart expt!\n");
    return __LINE__;
  }
  double maxerr = 0.0, inval = 0.0, outval = 0.0;
  for (size_t i = 0; i < len; i++)
    if (fabs(inbuf[i] - outbuf[i]) > maxerr) {
      inval = inbuf[i];
      outval = outbuf[i];
      maxerr = fabs(inbuf[i] - outbuf[i]);
    }
  printf("-- analysis: max error = %.2e (in = %e, out = %e)\n", maxerr, inval, outval);

  /* write out transformed data if needed */


  /* clean up allocated memory */
  free(meta);
  free(outbuf);
  free(inbuf);
}
