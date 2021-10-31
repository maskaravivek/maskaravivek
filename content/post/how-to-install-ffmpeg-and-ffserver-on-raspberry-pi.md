---
title: "How to install FFMPEG and FFServer on Raspberry Pi"
author: "Vivek Maskara"
date: 2020-10-14T08:05:26.367Z
lastmod: 2021-10-29T21:26:10-07:00

description: ""

subtitle: ""

categories: [Raspberry Pi]

tags:
 - FFmpeg
 - IoT
 - FFServer
 - Raspberry Pi

image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2020-10-14_how-to-install-ffmpeg-and-ffserver-on-raspberry-pi_0.jpeg"
 - "/post/img/2020-10-14_how-to-install-ffmpeg-and-ffserver-on-raspberry-pi_1.png"


aliases:
- "/how-to-install-ffmpeg-and-ffserver-on-raspberry-pi-ed0eddf86f88"

---

![](/post/img/2020-10-14_how-to-install-ffmpeg-and-ffserver-on-raspberry-pi_0.jpeg#layoutTextWidth)

Recently while working on a project, I needed to install FFmpeg on a Raspberry Pi. I also required FFServer along with FFmpeg. There are two issues here, firstly a standard binary of FFMpeg is not available for Raspberry, and secondly, FFServer is no longer packaged along with FFmpeg. It was removed after version 3.4.

This post illustrates all the steps for installing FFmpeg and FFServer on Raspberry Pi. You can get a [Raspberry Pi](https://amzn.to/3jJoeLC) from Amazon if you don’t already have one.

### Install prerequisites

First, we will install the prerequisites for the FFmpeg library. Execute the commands below to install the required libraries.

```
sudo apt updatesudo apt upgrade

sudo apt -y install autoconf automake build-essential cmake doxygen git graphviz imagemagick libasound2-dev libass-dev libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavutil-dev libfreetype6-dev libgmp-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libopus-dev librtmp-dev libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-net-dev libsdl2-ttf-dev libsnappy-dev libsoxr-dev libssh-dev libssl-dev libtool libv4l-dev libva-dev libvdpau-dev libvo-amrwbenc-dev libvorbis-dev libwebp-dev libx264-dev libx265-dev libxcb-shape0-dev libxcb-shm0-dev libxcb-xfixes0-dev libxcb1-dev libxml2-dev lzma-dev meson nasm pkg-config python3-dev python3-pip texinfo wget yasm zlib1g-dev libdrm-dev
```

### Install Add Ons

Next, we will be installing some other add on libraries that are required by FFmpeg. Some of these might not be required for your setup but for the sake of completeness, we will be installing all of these.

First, create a directory in which you can clone and build all these add ons.

```
mkdir ~/ffmpeg-libraries
```

Next, let's install `fdk-aac`

```
git clone --depth 1 https://github.com/mstorsjo/fdk-aac.git ~/ffmpeg-libraries/fdk-aac \
  && cd ~/ffmpeg-libraries/fdk-aac \
  && autoreconf -fiv \
  && ./configure \
  && make -j$(nproc) \
  && sudo make install
```

Next, let’s install `dav1d`

```
git clone --depth 1 https://code.videolan.org/videolan/dav1d.git ~/ffmpeg-libraries/dav1d \
  && mkdir ~/ffmpeg-libraries/dav1d/build \
  && cd ~/ffmpeg-libraries/dav1d/build \
  && meson .. \
  && ninja \
  && sudo ninja install
```

Next, let’s install `kvazaar`

```
git clone --depth 1 https://github.com/ultravideo/kvazaar.git ~/ffmpeg-libraries/kvazaar \
  && cd ~/ffmpeg-libraries/kvazaar \
  && ./autogen.sh \
  && ./configure \
  && make -j$(nproc) \
  && sudo make install
```

Next, let’s install `aom`:

```
git clone --depth 1 https://aomedia.googlesource.com/aom ~/ffmpeg-libraries/aom \
  && mkdir ~/ffmpeg-libraries/aom/aom_build \
  && cd ~/ffmpeg-libraries/aom/aom_build \
  && cmake -G "Unix Makefiles" AOM_SRC -DENABLE_NASM=on -DPYTHON_EXECUTABLE="$(which python3)" -DCMAKE_C_FLAGS="-mfpu=vfp -mfloat-abi=hard" .. \
  && sed -i 's/ENABLE_NEON:BOOL=ON/ENABLE_NEON:BOOL=OFF/' CMakeCache.txt \
  && make -j$(nproc) \
  && sudo make install
```

Finally, let’s install `zimg`

```
git clone https://github.com/sekrit-twc/zimg.git ~/ffmpeg-libraries/zimg \
  && cd ~/ffmpeg-libraries/zimg \
  && sh autogen.sh \
  && ./configure \
  && make \
  && sudo make install
```

Note: I have taken all these commands from [this](https://pimylifeup.com/compiling-ffmpeg-raspberry-pi/) blog post. The reason for writing a separate post is to have the latest post with all commands needed for both FFmpeg and FFserver.

Finally, let’s update the link cache:

```
sudo ldconfig
```

### Install FFmpeg

As mentioned earlier, FFmpeg no longer bundles FFserver. As stated in [this](https://superuser.com/questions/1296377/why-am-i-getting-an-unable-to-find-a-suitable-output-format-for-http-localho/1297419#1297419) thread, we need to check out an older version of FFmpeg to access FFserver.

Execute the following commands to clone and build `ffmpeg` from source.

```
git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg
cd ffmpeg
git checkout 2ca65fc7b74444edd51d5803a2c1e05a801a6023
./configure
```

Note:

- A pre-build library doesn’t exist for Raspberry Pi so the library needs to be built from source.
- It might take **2–3 hours**for FFmpeg to install. So be patient.
- If you are still getting `command not found` error, try restarting your Pi.

That’s it. Now you should have both FFmpeg and FFserver installed on your Raspberry Pi.

In my next post, I will show you how to use ffserver for streaming a live feed from a DSLR connected to the Raspberry Pi. Check out [this thread](http://gphoto-software.10949.n7.nabble.com/liveview-in-EOS-cameras-tp4238p13500.html) if you are trying to do something similar and need help with it.

You can buy me a coffee if this post really helped you learn something or fix a nagging issue!

* * *
Written on October 14, 2020 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/how-to-install-ffmpeg-and-ffserver-on-raspberry-pi-ed0eddf86f88)
