#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "MURaMKit_CAPI.h"

#define FLT float

int main(int argc, char** argv)
{
  char* infile = NULL;
  char* outfile = NULL;
  char* outmeta = NULL;

  size_t dim_fast = 0;
  size_t dim_mid = 0;
  size_t dim_slow = 0;

  if (argc == 5) {
    infile   = argv[1];
    dim_fast = atol(argv[2]);
    dim_mid  = atol(argv[3]);
    dim_slow = atol(argv[4]);
  }
  else if (argc == 7) {
    infile   = argv[1];
    dim_fast = atol(argv[2]);
    dim_mid  = atol(argv[3]);
    dim_slow = atol(argv[4]);
    outfile  = argv[5];
    outmeta  = argv[6];
  }
  else {
    printf("Usage: ./slice_norm input_file dim_fast dim_mid dim_slow [output_file]  [output_metadata]\n");
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
  else if (len != dim_fast * dim_mid * dim_slow) {
    printf("!! input file volume dimension error!\n");
    return __LINE__;
  }
  else
    printf("-- analysis: input has %ld values!\n", len);

  fseek(f, 0, SEEK_SET);
  FLT* inbuf = (FLT*)malloc(len * sizeof(FLT));
  size_t tmp = fread(inbuf, sizeof(FLT), len, f);
  assert(tmp == len);
  fclose(f);

  /* apply slice norm to a copy */
  FLT* outbuf = (FLT*)malloc(len * sizeof(FLT));
  memcpy(outbuf, inbuf, len * sizeof(FLT));
  void* meta = NULL;
  int rtn = mkit_slice_norm(outbuf, sizeof(FLT) == 4, dim_fast, dim_mid, dim_slow, &meta);
  if (rtn) {
    printf("!! error when applying slice normalization!\n");
    return __LINE__;
  }
  else
    printf("-- status: successfully applied slice normalization, meta size = %lu\n", 
            mkit_slice_norm_meta_len(meta));

  /* verification: apply inverse slice norm */
  rtn = mkit_inv_slice_norm(outbuf, sizeof(FLT) == 4, dim_fast, dim_mid, dim_slow, meta);
  if (rtn) {
    printf("!! error when applying inverse normalize!\n");
    return __LINE__;
  }
  else
    printf("-- status: successfully applied inverse slice normalization.\n");

  /* print out the maximum difference */
  double maxerr = 0.0, inval = 0.0, outval = 0.0;
  for (size_t i = 0; i < len; i++)
    if (fabs(inbuf[i] - outbuf[i]) > maxerr) {
      inval = inbuf[i];
      outval = outbuf[i];
      maxerr = fabs(inbuf[i] - outbuf[i]);
    }
  printf("-- analysis: max error = %.2e, rel = %.2e, (orig = %.2e, xform = %.2e)\n",
             maxerr, fabs(maxerr / inval), inval, outval);

  /* write out transformed data if needed */
  if (outfile && outmeta) {
    f = fopen(outfile, "w");
    fwrite(outbuf, sizeof(FLT), len, f);
    fclose(f);
    f = fopen(outmeta, "w");
    fwrite(meta, 1, mkit_log_meta_len(meta), f);
    fclose(f);
  }

  /* clean up allocated memory */
  if (meta)
    free(meta);
  free(outbuf);
  free(inbuf);
}
