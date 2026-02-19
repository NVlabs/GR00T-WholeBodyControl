#include "humanoid.hpp"

int main(int argc, char const *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " networkInterface" << std::endl;
    exit(-1);
  }

  HumanoidExample example(argv[1]);
  while (1) {
    sleep(10);
  }
  return 0;
}
