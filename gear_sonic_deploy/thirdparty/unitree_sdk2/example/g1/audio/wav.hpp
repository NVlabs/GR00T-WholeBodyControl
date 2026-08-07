#ifndef __UT_ROBOT_WAV_READER_HPP__
#define __UT_ROBOT_WAV_READER_HPP__

struct WaveHeader {
  void SeekToDataChunk(std::istream &is) {
    while (is && subchunk2_id != 0x61746164) {
      is.seekg(subchunk2_size, std::istream::cur);
      is.read(reinterpret_cast<char *>(&subchunk2_id), sizeof(int32_t));
      is.read(reinterpret_cast<char *>(&subchunk2_size), sizeof(int32_t));
    }
  }

  int32_t chunk_id;
  int32_t chunk_size;
  int32_t format;
  int32_t subchunk1_id;
  int32_t subchunk1_size;
  int16_t audio_format;
  int16_t num_channels;
  int32_t sample_rate;
  int32_t byte_rate;
  int16_t block_align;
  int16_t bits_per_sample;
  int32_t subchunk2_id;    // a tag of this chunk
  int32_t subchunk2_size;  // size of subchunk2
};

static_assert(sizeof(WaveHeader) == 44);

std::vector<uint8_t> ReadWaveImpl(std::istream &is, int32_t *sampling_rate,
                                  int8_t *channelCount, bool *is_ok) {
  WaveHeader header{};
  is.read(reinterpret_cast<char *>(&header.chunk_id), sizeof(header.chunk_id));

  //                        F F I R
  if (header.chunk_id != 0x46464952) {
    printf("Expected chunk_id RIFF. Given: 0x%08x\n", header.chunk_id);
    *is_ok = false;
    return {};
  }

  is.read(reinterpret_cast<char *>(&header.chunk_size),
          sizeof(header.chunk_size));

  is.read(reinterpret_cast<char *>(&header.format), sizeof(header.format));

  //                      E V A W
  if (header.format != 0x45564157) {
    printf("Expected format WAVE. Given: 0x%08x\n", header.format);
    *is_ok = false;
    return {};
  }

  is.read(reinterpret_cast<char *>(&header.subchunk1_id),
          sizeof(header.subchunk1_id));

  is.read(reinterpret_cast<char *>(&header.subchunk1_size),
          sizeof(header.subchunk1_size));

  if (header.subchunk1_id == 0x4b4e554a) {
    // skip junk padding
    is.seekg(header.subchunk1_size, std::istream::cur);

    is.read(reinterpret_cast<char *>(&header.subchunk1_id),
            sizeof(header.subchunk1_id));

    is.read(reinterpret_cast<char *>(&header.subchunk1_size),
            sizeof(header.subchunk1_size));
  }

  if (header.subchunk1_id != 0x20746d66) {
    printf("Expected subchunk1_id 0x20746d66. Given: 0x%08x\n",
           header.subchunk1_id);
    *is_ok = false;
    return {};
  }

  if (header.subchunk1_size != 16 &&
      header.subchunk1_size != 18) {  // 16 for PCM
    printf("Expected subchunk1_size 16. Given: %d\n", header.subchunk1_size);
    *is_ok = false;
    return {};
  }

  is.read(reinterpret_cast<char *>(&header.audio_format),
          sizeof(header.audio_format));

  if (header.audio_format != 1) {  // 1 for PCM
    printf("Expected audio_format 1. Given: %d\n", header.audio_format);
    *is_ok = false;
    return {};
  }

  is.read(reinterpret_cast<char *>(&header.num_channels),
          sizeof(header.num_channels));

  *channelCount = (int8_t)header.num_channels;

  is.read(reinterpret_cast<char *>(&header.sample_rate),
          sizeof(header.sample_rate));

  is.read(reinterpret_cast<char *>(&header.byte_rate),
          sizeof(header.byte_rate));

  is.read(reinterpret_cast<char *>(&header.block_align),
          sizeof(header.block_align));

  is.read(reinterpret_cast<char *>(&header.bits_per_sample),
          sizeof(header.bits_per_sample));

  if (header.byte_rate !=
      (header.sample_rate * header.num_channels * header.bits_per_sample / 8)) {
    printf("Incorrect byte rate: %d. Expected: %d", header.byte_rate,
           (header.sample_rate * header.num_channels * header.bits_per_sample /
            8));
    *is_ok = false;
    return {};
  }

  if (header.block_align !=
      (header.num_channels * header.bits_per_sample / 8)) {
    printf("Incorrect block align: %d. Expected: %d\n", header.block_align,
           (header.num_channels * header.bits_per_sample / 8));
    *is_ok = false;
    return {};
  }

  if (header.bits_per_sample != 16) {  // we support only 16 bits per sample
    printf("Expected bits_per_sample 16. Given: %d\n", header.bits_per_sample);
    *is_ok = false;
    return {};
  }

  if (header.subchunk1_size == 18) {
    int16_t extra_size = -1;
    is.read(reinterpret_cast<char *>(&extra_size), sizeof(int16_t));
    if (extra_size != 0) {
      printf(
          "Extra size should be 0 for wave from NAudio. Current extra size "
          "%d\n",
          extra_size);
      *is_ok = false;
      return {};
    }
  }

  is.read(reinterpret_cast<char *>(&header.subchunk2_id),
          sizeof(header.subchunk2_id));

  is.read(reinterpret_cast<char *>(&header.subchunk2_size),
          sizeof(header.subchunk2_size));

  header.SeekToDataChunk(is);
  if (!is) {
    *is_ok = false;
    return {};
  }

  *sampling_rate = header.sample_rate;

  // header.subchunk2_size contains the number of bytes in the data.
  // As we assume each sample contains two bytes, so it is divided by 2 here
  std::vector<int16_t> samples(header.subchunk2_size / 2);

  is.read(reinterpret_cast<char *>(samples.data()), header.subchunk2_size);
  if (!is) {
    *is_ok = false;
    return {};
  }

  std::vector<uint8_t> ans(samples.size() * 2);
  for (int32_t i = 0; i != static_cast<int32_t>(samples.size()); ++i) {
    ans[i * 2] = samples[i] & 0xFF;
    ans[i * 2 + 1] = (samples[i] >> 8) & 0xFF;
  }

  *is_ok = true;
  return ans;
}

std::vector<uint8_t> ReadWave(const std::string &filename,
                              int32_t *sampling_rate, int8_t *channelCount,
                              bool *is_ok) {
  std::ifstream is(filename, std::ifstream::binary);
  auto samples = ReadWaveImpl(is, sampling_rate, channelCount, is_ok);
  return samples;
}

bool WriteWave(const std::string &filename, int32_t sampling_rate,
               const int16_t *samples, int32_t n, uint8_t num_channels) {
  WaveHeader header{};
  header.chunk_id = 0x46464952;      // FFIR
  header.format = 0x45564157;        // EVAW
  header.subchunk1_id = 0x20746d66;  // "fmt "
  header.subchunk1_size = 16;        // 16 for PCM
  header.audio_format = 1;           // PCM =1

  int32_t bits_per_sample = 16;  // int16_t
  header.num_channels = num_channels;
  header.sample_rate = sampling_rate;
  header.byte_rate = sampling_rate * num_channels * bits_per_sample / 8;
  header.block_align = num_channels * bits_per_sample / 8;
  header.bits_per_sample = bits_per_sample;
  header.subchunk2_id = 0x61746164;  // atad
  header.subchunk2_size = n * num_channels * bits_per_sample / 8;

  header.chunk_size = 36 + header.subchunk2_size;

  std::vector<int16_t> samples_int16(n * num_channels);
  for (int32_t i = 0; i != n * num_channels; ++i) {
    samples_int16[i] = samples[i];
  }

  std::ofstream os(filename, std::ios::binary);
  if (!os) {
    printf("Failed to create %s", filename.c_str());
    return false;
  }

  os.write(reinterpret_cast<const char *>(&header), sizeof(header));
  os.write(reinterpret_cast<const char *>(samples_int16.data()),
           samples_int16.size() * sizeof(int16_t));

  if (!os) {
    printf("Write %s failed", filename.c_str());
    return false;
  }

  return true;
}

#endif
