# Raspberry Pi Real Time Object Detection

## RUN

```
xhost +
```

```
docker run -it --rm -e DISPLAY=$DISPLAY \
       -v /tmp/.X11-unix:/tmp/.X11-unix \
       --device /dev/video0 raspi-realtime
```
