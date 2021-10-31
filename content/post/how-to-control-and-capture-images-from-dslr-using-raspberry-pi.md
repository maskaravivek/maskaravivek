---
title: "How to Control and Capture Images from DSLR using Raspberry Pi"
author: "Vivek Maskara"
date: 2020-10-17T07:27:48.966Z
lastmod: 2021-10-29T21:26:10-07:00

description: ""

subtitle: ""

categories: [Raspberry Pi]

tags:
 - Raspberry Pi
 - IoT
 - DSLR
 - Gphoto2

image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2020-10-17_how-to-control-and-capture-images-from-dslr-using-raspberry-pi_0.jpeg"


aliases:
- "/how-to-control-and-capture-images-from-dslr-using-raspberry-pi-fdfa9d600ec1"

---

![](/post/img/2020-10-17_how-to-control-and-capture-images-from-dslr-using-raspberry-pi_0.jpeg#layoutTextWidth)

In this short post, I will walk you through the steps for controlling and capturing pictures from a DSLR connected to the Raspberry Pi using a USB cable. There are numerous tutorials out there for this setup but I found them to be outdated to some extent. This post covers the installation of libgphoto2 and [ghoto2](http://www.gphoto.org/) from the source and also covers the steps for capturing pictures using Python scripts.

### Prerequisite

For this tutorial, I am assuming that you have a Raspberry Pi with Raspbian or Noob OS installed on it. You can get a [Raspberry Pi](https://amzn.to/3jJoeLC) from Amazon if you don’t already have one.

Also, I am assuming that you already have one of the supported cameras listed here:

[Projects :: libgphoto2 :: supported cameras](http://www.gphoto.org/proj/libgphoto2/support.php "http://www.gphoto.org/proj/libgphoto2/support.php")

I used the [Canon Rebel T7 camera](https://amzn.to/3jIBccL) which I got from Amazon.

### Install libgphoto2 and gphoto2

`libghoto2` library lets you interface with 100s of supported DSLR cameras and `ghoto2` is a command-line utility for using `libghoto2`

Here are the steps for installing both these libraries.

#### Prerequisites

Install the following dependencies:

```
sudo apt-get install git make autoconf libltdl-dev libusb-dev libexif-dev libpopt-dev libxml2-dev libjpeg-dev libgd-dev gettext autopoint
```

#### Install libghoto2

Download the latest code for `libgphoto2` from:

```
git clone https://github.com/gphoto/libgphoto2.git
```

Execute the following commands to install `libgphoto2`

```
cd ~/libgphoto2
autoreconf --install --symlink
./configure
make
sudo make install
```

#### Install ghoto2

Download code for `gphoto2`

```
git clone https://github.com/gphoto/gphoto2.git
```

Build and install`gphoto2`

```
cd ~/gphoto2
autoreconf --install --symlink
./configure
make
sudo make install
```

#### Update Configs

Add the following line in `/etc/ld.so.conf.d/libc.conf`

```
/usr/local/lib
```

Refresh cache before proceeding further.

```
sudo ldconfig
```

Generate `udev` rules for the camera

```
/usr/local/lib/libgphoto2/print-camera-list udev-rules version 201 group plugdev mode 0660 | sudo tee /etc/udev/rules.d/90-libgphoto2.rules
```

Generate the hardware database file for `udev`

```
/usr/local/lib/libgphoto2/print-camera-list hwdb | sudo tee /etc/udev/hwdb.d/20-gphoto.hwdb
```

### Using ghoto2

Now that we have both the libraries installed, now we can use `ghoto2` for interfacing with the camera. Make sure that the camera is connected to the Raspberry Pi.

Execute the following command to see if `ghoto2` can detect the camera.

```
gphoto2 --auto-detect
```

If your camera name shows up in the output, you can now go ahead and click a picture by executing the following command:

```
gphoto2 --capture-image
```

### Using Python Scripts

Now that we have tested that the camera connection is working fine, we will go ahead and capture pictures using simple python scripts. Firstly, install the python wrapper for `ghoto2`

```
sudo pip install -v gphoto2
```

Next, create a script that will let you capture an image from the DSLR.

```
import logging
import os
import subprocess
import sys

import gphoto2 as gp

logging.basicConfig(
        format='%(levelname)s: %(name)s: %(message)s', level=logging.WARNING)
callback_obj = gp.check_result(gp.use_python_logging())

def get_camera():
    camera = gp.Camera()
    camera.init()
    return camera

def capture_image_from_dslr():
    camera = get_camera()
    capture_image(camera)
    camera.exit()
    return 0

def capture_image(camera):
    print('Capturing image')
    file_path = camera.capture(gp.GP_CAPTURE_IMAGE)
    print('Camera file path: {0}/{1}'.format(file_path.folder, file_path.name))
    target = os.path.join('/tmp', file_path.name)
    print('Copying image to', target)
    camera_file = camera.file_get(
        file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL)
    camera_file.save(target)
    return 0
  
  if __name__ == "__main__":
    sys.exit(capture_image_from_dslr())
```

We first get an instance of the camera object using `get_camera` and then pass this instance to the`capture_image` function to click a picture.

That’s it for this post. I hope you would have found this useful.

* * *
Written on October 17, 2020 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/how-to-control-and-capture-images-from-dslr-using-raspberry-pi-fdfa9d600ec1)
