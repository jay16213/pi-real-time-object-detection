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
ADD ./tensorflow-1.1.0-cp35-cp35m-linux_armv7l.whl .
RUN pip3 install ./tensorflow-1.1.0-cp35-cp35m-linux_armv7l.whl
#RUN pip3 install tensorflow
#RUN pip3 install Pillow

RUN pip3 install keras==2.1.2

# RUN apt-get install python-picamera python3-picamera

# RUN apt-get install rpi-update
# RUN rpi-update
# RUN modprobe bcm2835-v4l2

# yad2k
COPY ./src/ /app
WORKDIR /app/

# RUN apt-get install xinit
# RUN apt-get install xserver-xorg
# RUN apt-get install xserver-xorg-video-fbdev
#RUN xeyes -display :0.0
CMD ["python3", "pi-realtime.py"]

