# Raspberry Pi Real Time Object Detection

## Run with Docker

Install docker if you have not install on pi yet
```bash
curl -fsSL https://get.docker.com/ | sh
```

If you would like to use Docker as a non-root user, add your user to the "docker" group
```bash
sudo usermod -aG docker <user>
```

After usermod you should logout to save the change


```bash
docker run -it --rm --privileged -v ./src/:/app \
       --device /dev/video0 raspi-realtime
```
