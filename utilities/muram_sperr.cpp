#include <iostream>
#include <string>

#include "MURaMKit.h"

#include "SPERR3D_OMP_C.h"
#include "SPERR3D_OMP_D.h"

int main(int argc, char* argv[])
{
  if (argc != 6) {
    std::cout << "Usage: ./muram_sperr  infile  dimx  dimy  dimz  outfile" << std::endl;
    return 0;
  }

  const auto infile = std::string(argv[1]);
  const auto outfile = std::string(argv[5]);
  const size_t dimx = std::stoi(argv[2]);
  const size_t dimy = std::stoi(argv[3]);
  const size_t dimz = std::stoi(argv[4]);
  const auto total_len = dimx * dimy * dimz;

  // Read input file
  auto inbuf = sperr::read_whole_file<float>(infile);
  if (inbuf.size() != total_len) {
    std::cout << "Read input file wrong!" << std::endl;
    return __LINE__;
  }

  // Conditioning meta data
  void* meta = NULL;
  size_t meta_size = 0;

  // Apply smart log transform
  auto rtni = mkit::smart_log(inbuf.data(), inbuf.size(), &meta);
  if (!meta || rtni) {
    std::cout << "pre-conditioning failed!" << std::endl;
    return __LINE__;
  }
  meta_size = mkit::retrieve_log_meta_len(meta);

  // Apply SPERR compression
  auto encoder = std::make_unique<sperr::SPERR3D_OMP_C>();
  encoder->set_dims_and_chunks({dimx, dimy, dimz}, {288, 256, 256});
  encoder->set_num_threads(6);
  encoder->set_psnr(130.0);
  auto rtn = encoder->compress(inbuf.data(), inbuf.size());
  if (rtn != sperr::RTNType::Good) {
    std::cout << "SPERR compression failed!" << std::endl;
    std::free(meta);
    return __LINE__;
  }
  auto stream = encoder->get_encoded_bitstream();
  encoder.reset();

  // Apply SPERR decompression
  auto decoder = std::make_unique<sperr::SPERR3D_OMP_D>();
  decoder->set_num_threads(6);
  decoder->setup_decomp(stream.data(), stream.size());
  rtn = decoder->decompress(stream.data());
  if (rtn != sperr::RTNType::Good) {
    std::cout << "SPERR decompression failed!" << std::endl;
    std::free(meta);
    return __LINE__;
  }

  // Acquire the decompressed data
  auto outbufd = decoder->release_decoded_data();
  decoder.reset();
  auto outbuff = std::vector<float>(total_len);
  std::copy(outbufd.begin(), outbufd.end(), outbuff.begin());
  outbufd.clear();
  outbufd.shrink_to_fit();

  // Apply smart exp transform
  rtni = mkit::smart_exp(outbuff.data(), total_len, meta);
  if (rtni) {
    std::cout << "post-conditioning failed!" << std::endl;
    std::free(meta);
    return __LINE__;
  }

  // Write output file
  rtn = sperr::write_n_bytes(outfile, sizeof(float) * outbuff.size(), outbuff.data());
  if (rtn != sperr::RTNType::Good) {
    std::cout << "Output decompression file failed!" << std::endl;
    std::free(meta);
    return __LINE__;
  }

  if (meta)
    std::free(meta);

  return 0;
}
