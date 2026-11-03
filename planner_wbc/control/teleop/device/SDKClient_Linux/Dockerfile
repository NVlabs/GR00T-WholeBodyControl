# Copyright 2018 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.#
#
# Based on https://hub.docker.com/r/grpc/cxx.

# To build
#   docker build -f ./Dockerfile -t manus-linux .

# To Run (Linux)
#   docker run --net==host --privileged -v /dev:/dev -v /run/udev:/run/udev -i -t manus-linux /bin/bash

# To Run (Windows)
#   docker run -p 5000:5000 -i -t manus-linux /bin/bash

# FROM ubuntu:jammy as build
FROM ubuntu:focal as build
LABEL description="Visual Studio Manus SDK Build container" 

ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
  build-essential \
  # Required to install GRPC
  git \
  libtool \
  libzmq3-dev \
  # Required for core integrated
  libusb-1.0-0-dev \
  libudev-dev \
  # Only required for building the minimal client
  libncurses5-dev \ 
  # Only required for visual studio debugging
  gdb \
  && apt-get clean

ENV GRPC_RELEASE_TAG="v1.28.1"

RUN git clone -b ${GRPC_RELEASE_TAG} https://github.com/grpc/grpc /var/local/git/grpc && \
    cd /var/local/git/grpc && \
    git submodule update --init --recursive

RUN echo "-- installing protobuf" && \
    cd /var/local/git/grpc/third_party/protobuf && \
    ./autogen.sh && ./configure --enable-shared && \
    make -j$(nproc) && make -j$(nproc) check && make install && make clean && ldconfig

RUN echo "-- installing grpc" && \
    cd /var/local/git/grpc && \
    make -j$(nproc) && make install && make clean && ldconfig

# configure SSH for communication with Visual Studio 
RUN apt-get update && apt-get install -y openssh-server
RUN mkdir -p /var/run/sshd
RUN echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config && \ 
   ssh-keygen -A 

# please make sure to change the password
RUN id -u manus >/dev/null 2>&1 || useradd -m -d /home/manus -s /bin/bash -G sudo manus && \
echo "manus:password" | chpasswd

# copy dependencies to ~/ManusSDK/ directory
COPY ManusSDK /home/manus/ManusSDK

# Add read/write permissions for manus devices
RUN apt-get update && apt-get install -y udev
RUN echo "# HIDAPI/libusb" > /etc/udev/rules.d/99-manus.rules && \
    echo "SUBSYSTEMS==\"usb\", ATTRS{idVendor}==\"3325\", MODE:=\"0666\"" >> /etc/udev/rules.d/99-manus.rules && \
    echo "# HIDAPI/hidraw" >> /etc/udev/rules.d/99-manus.rules && \
    echo "KERNEL==\"hidraw*\", ATTRS{idVendor}==\"3325\", MODE:=\"0666\"" >> /etc/udev/rules.d/99-manus.rules

# change default SSH port to 5000 to not conflict with system ssh in host mode
RUN echo "Port 5000" >> /etc/ssh/sshd_config

ENTRYPOINT service ssh start && service udev start && bin/bash

EXPOSE 5000
