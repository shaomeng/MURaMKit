#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <memory>

#include "Bitmask.h"

using FLT = float;

int main(int argc, char* argv[])
{
  char* infile = nullptr;
  char* outfile = nullptr;
  char* outmeta = nullptr;

  if (argc == 4) {
    infile = argv[1];
    outfile = argv[2];
    outmeta = argv[3];
  }
  else {
    std::printf("Usage: ./bitmask  input_file  output_file  output_mask\n");
    return __LINE__;
  }

  // read infile
  FILE* f = std::fopen(infile, "r");
  if (!f) {
    std::printf("!! input file doesn't exist: %s\n", infile);
    return __LINE__;
  }
  fseek(f, 0, SEEK_END);
  const long len = std::ftell(f) / sizeof(FLT);  // number of floats/doubles
  if (std::ftell(f) % sizeof(FLT) != 0) {
    std::printf("!! input file size error!\n");
    return __LINE__;
  }
  else
    std::printf("-- analysis: input has %ld values!\n", len);

  std::fseek(f, 0, SEEK_SET);
  auto inbuf = std::make_unique<FLT[]>(len);
  size_t tmp = std::fread(inbuf.get(), sizeof(FLT), len, f);
  assert(tmp == len);
  std::fclose(f);

  // Separate non-zero values.
  auto mask = mkit::Bitmask(len);
  mask.reset();
  auto nonzero = std::vector<FLT>(len / 16);
  for (size_t i = 0; i < len; i++) {
    if (std::abs(inbuf[i]) < 1e-11)
      mask.write_true(i);
    else
      nonzero.push_back(inbuf[i]);
  }
  const auto& mask_buf = mask.view_buffer();

  // write out transformed data
  f = std::fopen(outfile, "w");
  std::fwrite(nonzero.data(), sizeof(FLT), nonzero.size(), f);
  std::fclose(f);
  f = std::fopen(outmeta, "w");
  std::fwrite(mask_buf.data(), sizeof(long), mask_buf.size(), f);
  std::fclose(f);

}
