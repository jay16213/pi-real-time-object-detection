FROM resin/rpi-raspbian:stretch
LABEL maintainer="jay101630@gmail.com"

# Uncomment the this line if you want to cross build this image
# Follow the tutorials to build the image on intel-based host (e.g. Ubuntu on intel CPU)
# Reference: http://www.hotblackrobotics.com/en/blog/2018/01/22/docker-images-arm/
# You may get an unexpected error when you build image with qemu
# COPY ./qemu-arm-static /usr/bin/qemu-arm-static


# Install the libary
RUN apt-get update
RUN apt-get upgrade
RUN apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-dev \
    libblas-dev \
    liblapack-dev \
    libhdf5-dev \
    python3-h5py \
    python3-scipy  \
    python3-pil


# pi camera modules
RUN apt-get install --reinstall libraspberrypi0 libraspberrypi-bin
RUN pip3 install picamera

# Instal numpy and Pillow
# RUN pip3 install numpy Pillow

# The latest keras has bug, use 2.1.2 instead
# See this issue here: https://github.com/keras-team/keras/issues/9349
RUN pip3 install keras==2.1.2


# Install OpenCV 3.4.0
WORKDIR /
RUN wget https://github.com/opencv/opencv/archive/3.4.0.zip && unzip 3.4.0.zip
RUN mkdir /opencv-3.4.0/build \
&& cd /opencv-3.4.0/build \
&& cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local ..
WORKDIR /opencv-3.4.0/build
RUN make -j4 && make install
WORKDIR /
RUN rm -rf /opencv-3.4.0 \
&& rm 3.4.0.zip


# Install tensorflow, use unofficial binaries
# Repo: https://github.com/samjabrahams/tensorflow-on-raspberry-pi.git
RUN wget https://github.com/samjabrahams/tensorflow-on-raspberry-pi/releases/download/v1.1.0/tensorflow-1.1.0-cp34-cp34m-linux_armv7l.whl
RUN mv tensorflow-1.1.0-cp34-cp34m-linux_armv7l.whl tensorflow-1.1.0-cp35-cp35m-linux_armv7l.whl
RUN pip3 install tensorflow-1.1.0-cp35-cp35m-linux_armv7l.whl
RUN rm tensorflow-1.1.0-cp35-cp35m-linux_armv7l.whl


# copy source code
COPY ./src /app
WORKDIR /app

CMD ["python3", "pi-realtime.py"]
