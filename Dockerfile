FROM raspi-opencv
LABEL maintainer="jay101630@gmail.com"

# COPY ./qemu-arm-static /usr/bin/qemu-arm-static

# RUN apt-get update
# # RUN apt-get upgrade

# RUN apt-get install \
#     python3 \
#     python3-pip \
#     python3-setuptools \
#     python3-dev \
#     libblas-dev \
#     liblapack-dev \
#     libhdf5-dev \
#     python3-h5py \
#     python3-scipy  \
#     python3-pil

# RUN pip3 install numpy keras

# install OpenCV 3.4.0
# WORKDIR /
# RUN wget https://github.com/opencv/opencv/archive/3.4.0.zip && unzip 3.4.0.zip
# RUN mkdir /opencv-3.4.0/build \
# && cd /opencv-3.4.0/build \
# && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local .. \
# && make -j4 \
# && make install

# install tensorflow
RUN pip3 install tensorflow
RUN pip3 install Pillow

# yad2k
COPY ./src/ /app
WORKDIR /app/yad2k
CMD ["python3", "yad2k-realtime.py"]

