#include <iostream>
#include <string>

#include "MURaMKit.h"

#include "SPERR3D_OMP_C.h"
#include "SPERR3D_OMP_D.h"

#define SLICE_NORM

int main(int argc, char* argv[])
{
  if (argc < 6) {
    std::cout << "Usage: ./muram_sperr  infile  dimx  dimy  dimz  outfile [psnr]" << std::endl;
    return 0;
  }

  const auto infile = std::string(argv[1]);
  const auto outfile = std::string(argv[5]);
  const size_t dimx = std::stoi(argv[2]);
  const size_t dimy = std::stoi(argv[3]);
  const size_t dimz = std::stoi(argv[4]);
  const auto total_len = dimx * dimy * dimz;
  auto psnr = 130.0;
  if (argc = 7)
    psnr = std::stod(argv[6]);

  // Read input file
  auto inbuf = sperr::read_whole_file<float>(infile);
  if (inbuf.size() != total_len) {
    std::cout << "Read input file wrong!" << std::endl;
    std::printf("  Expecting %lu floats, got %lu floats\n", total_len, inbuf.size());
    return __LINE__;
  }

  // Declare conditioning meta data
  void* meta = NULL;
  size_t meta_size = 0;

#ifdef SMART_LOG
  // Apply smart log transform
  auto rtni = mkit::smart_log(inbuf.data(), inbuf.size(), &meta);
  if (!meta || rtni) {
    std::cout << "pre-conditioning failed!" << std::endl;
    return __LINE__;
  }
  meta_size = mkit::retrieve_log_meta_len(meta);
  auto b8 = mkit::unpack_8_booleans(reinterpret_cast<uint8_t*>(meta)[8]);
  if (b8[0])
    std::cout << "(Used a neg value mask)" << std::endl;
  if (b8[1])
    std::cout << "(Used a zero value mask)" << std::endl;
#endif

#ifdef SLICE_NORM
  // Apply slice-based normalization
  auto rtni = mkit::slice_norm(inbuf.data(), {dimx, dimy, dimz}, &meta);
  if (!meta || rtni) {
    std::cout << "pre-conditioning failed!" << std::endl;
    return __LINE__;
  }
  meta_size = mkit::retrieve_slice_norm_meta_len(meta);
#endif

  // Apply SPERR compression
  auto encoder = std::make_unique<sperr::SPERR3D_OMP_C>();
  encoder->set_dims_and_chunks({dimx, dimy, dimz}, {288, 256, 256});
  encoder->set_num_threads(7);
  encoder->set_psnr(psnr);
  auto rtn = encoder->compress(inbuf.data(), inbuf.size());
  if (rtn != sperr::RTNType::Good) {
    std::cout << "SPERR compression failed!" << std::endl;
    std::free(meta);
    return __LINE__;
  }
  auto stream = encoder->get_encoded_bitstream();
  encoder.reset();
  auto comp_size = stream.size() + meta_size;
  std::cout << "Compression bitrate: " << 8.0 * comp_size / total_len << std::endl;

  // Apply SPERR decompression
  auto decoder = std::make_unique<sperr::SPERR3D_OMP_D>();
  decoder->set_num_threads(7);
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

#ifdef SMART_LOG
  // Apply smart exp transform
  rtni = mkit::smart_exp(outbufd.data(), total_len, meta);
  if (rtni) {
    std::cout << "post-conditioning failed!" << std::endl;
    std::free(meta);
    return __LINE__;
  }
#endif

#ifdef SLICE_NORM
  // Apply inverse slice-based normalization
  rtni = mkit::inv_slice_norm(outbufd.data(), {dimx, dimy, dimz}, meta);
  if (rtni) {
    std::cout << "post-conditioning failed!" << std::endl;
    std::free(meta);
    return __LINE__;
  }
#endif

  // Make a copy of the data in single precision
  auto outbuff = std::vector<float>(total_len);
  std::copy(outbufd.begin(), outbufd.end(), outbuff.begin());

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
