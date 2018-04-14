#!/bin/bash

# Open x11 access control so that docker can connect to host
xhost +

# Simulate pi camera as a normal usb webcam
# You should install v4l2 driver first
# Tutorials: https://www.raspberrypi.org/forums/viewtopic.php?t=62364
sudo modprobe bcm2835-v4l2

# run docker
docker run -it --rm --privileged \
    -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY \
    raspi-realtime

