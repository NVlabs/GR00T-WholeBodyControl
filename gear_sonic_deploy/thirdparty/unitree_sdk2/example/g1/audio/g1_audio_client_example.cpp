#include <fstream>
#include <iostream>
#include <thread>
#include <unitree/common/time/time_tool.hpp>
#include <unitree/idl/ros2/String_.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/g1/audio/g1_audio_client.hpp>

#include "wav.hpp"

#define AUDIO_FILE_PATH "../example/g1/audio/test.wav"
#define AUDIO_SUBSCRIBE_TOPIC "rt/audio_msg"
#define GROUP_IP "239.168.123.161"
#define PORT 5555

#define WAV_SECOND 5  // record seconds
#define WAV_LEN (16000 * 2 * WAV_SECOND)
#define CHUNK_SIZE 96000  // 3 seconds
int sock;

void asr_handler(const void *msg) {
  std_msgs::msg::dds_::String_ *resMsg = (std_msgs::msg::dds_::String_ *)msg;
  std::cout << "Topic:\"rt/audio_msg\" recv: " << resMsg->data() << std::endl;
}

std::string get_local_ip_for_multicast() {
  struct ifaddrs *ifaddr, *ifa;
  char host[NI_MAXHOST];
  std::string result = "";

  getifaddrs(&ifaddr);
  for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (!ifa->ifa_addr || ifa->ifa_addr->sa_family != AF_INET) continue;
    getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in), host, NI_MAXHOST,
                NULL, 0, NI_NUMERICHOST);
    std::string ip(host);
    if (ip.find("192.168.123.") == 0) {
      result = ip;
      break;
    }
  }
  freeifaddrs(ifaddr);
  return result;
}

void thread_mic(void) {
  sock = socket(AF_INET, SOCK_DGRAM, 0);
  sockaddr_in local_addr{};
  local_addr.sin_family = AF_INET;
  local_addr.sin_port = htons(PORT);
  local_addr.sin_addr.s_addr = INADDR_ANY;
  bind(sock, (sockaddr *)&local_addr, sizeof(local_addr));

  ip_mreq mreq{};
  inet_pton(AF_INET, GROUP_IP, &mreq.imr_multiaddr);
  std::string local_ip = get_local_ip_for_multicast();
  std::cout << "local ip: " << local_ip << std::endl;
  mreq.imr_interface.s_addr = inet_addr(local_ip.c_str());
  setsockopt(sock, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq));

  int total_bytes = 0;
  std::vector<int16_t> pcm_data;
  pcm_data.reserve(WAV_LEN / 2);
  std::cout << "start record!" << std::endl;
  while (total_bytes < WAV_LEN) {
    char buffer[2048];
    ssize_t len = recvfrom(sock, buffer, sizeof(buffer), 0, nullptr, nullptr);
    if (len > 0) {
      size_t sample_count = len / 2;
      const int16_t *samples = reinterpret_cast<const int16_t *>(buffer);
      pcm_data.insert(pcm_data.end(), samples, samples + sample_count);
      total_bytes += len;
    }
  }

  WriteWave("record.wav", 16000, pcm_data.data(), pcm_data.size(), 1);
  std::cout << "record finish! save to record.wav " << std::endl;
}

int main(int argc, char const *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: audio_client_example [NetWorkInterface(eth0)]"
              << std::endl;
    exit(0);
  }
  int32_t ret;
  /*
   * Initilaize ChannelFactory
   */
  unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
  unitree::robot::g1::AudioClient client;
  client.Init();
  client.SetTimeout(10.0f);

  /*ASR message Example*/
  unitree::robot::ChannelSubscriber<std_msgs::msg::dds_::String_> subscriber(
      AUDIO_SUBSCRIBE_TOPIC);
  subscriber.InitChannel(asr_handler);

  /*Volume Example*/
  uint8_t volume;
  ret = client.GetVolume(volume);
  std::cout << "GetVolume API ret:" << ret
            << "  volume = " << std::to_string(volume) << std::endl;
  ret = client.SetVolume(100);
  std::cout << "SetVolume to 100% , API ret:" << ret << std::endl;

  /*TTS Example*/
  ret = client.TtsMaker("你好。我是宇树科技的机器人。例程启动成功",
                        0);  // Auto play
  std::cout << "TtsMaker API ret:" << ret << std::endl;
  unitree::common::Sleep(5);

  ret = client.TtsMaker(
      "Hello. I'm a robot from Unitree Robotics. The example has started "
      "successfully. ",
      1);  // Engilsh TTS
  std::cout << "TtsMaker API ret:" << ret << std::endl;
  unitree::common::Sleep(8);

  /*Audio Play Example*/
  int32_t sample_rate = -1;
  int8_t num_channels = 0;
  bool filestate = false;
  std::vector<uint8_t> pcm =
      ReadWave(AUDIO_FILE_PATH, &sample_rate, &num_channels, &filestate);

  std::cout << "wav file sample_rate = " << sample_rate
            << " num_channels =  " << std::to_string(num_channels)
            << " filestate =" << filestate << "filesize = " << pcm.size()
            << std::endl;

  if (filestate && sample_rate == 16000 && num_channels == 1) {
    size_t total_size = pcm.size();
    size_t offset = 0;
    int chunk_index = 0;
    std::string stream_id =
        std::to_string(unitree::common::GetCurrentTimeMillisecond());

    while (offset < total_size) {
      size_t remaining = total_size - offset;
      size_t current_chunk_size =
          std::min(static_cast<size_t>(CHUNK_SIZE), remaining);
      std::vector<uint8_t> chunk(pcm.begin() + offset,
                                 pcm.begin() + offset + current_chunk_size);
      client.PlayStream("example", stream_id, chunk);
      unitree::common::Sleep(1);
      std::cout << "Playing size: " << offset << std::endl;
      offset += current_chunk_size;
    }

    ret = client.PlayStop(stream_id);  // stop playback after transmission ends

  } else {
    std::cout << "audio file format error, please check!" << std::endl;
  }

  /*LED Control Example*/
  client.LedControl(0, 255, 0);
  unitree::common::Sleep(1);
  client.LedControl(0, 0, 0);
  unitree::common::Sleep(1);
  client.LedControl(0, 0, 255);

  std::cout << "AudioClient api test finish , asr start..." << std::endl;

  std::thread mic_t(thread_mic);

  while (1) {
    sleep(1);  // wait for asr message
  }
  mic_t.join();
  return 0;
}