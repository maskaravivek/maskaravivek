---
title: "How to install FFMPEG on EC2 running Amazon Linux?"
author: "Vivek Maskara"
date: 2019-10-21T19:14:34.293Z
lastmod: 2021-10-29T21:24:26-07:00

description: ""

subtitle: ""

categories: [AWS]

tags:
 - AWS
 - FFmpeg

image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2019-10-21_how-to-install-ffmpeg-on-ec2-running-amazon-linux_0.jpeg"
 - "/post/img/2019-10-21_how-to-install-ffmpeg-on-ec2-running-amazon-linux_1.png"


aliases:
- "/how-to-install-ffmpeg-on-ec2-running-amazon-linux-451e4a8e2694"

---

![](/post/img/2019-10-21_how-to-install-ffmpeg-on-ec2-running-amazon-linux_0.jpeg#layoutTextWidth)

Recently, I needed to install `ffmpeg` on my EC2 instance and I struggled quite a bit to set it up. The issue was that my EC2 instance is running Amazon Linux based AMI which is probably based on some version of CentOS. For Debian and Ubuntu distributions, `ffmpeg` is available as a `apt-get` package but for other distributions you have to manually compile it. To be fair `ffmpeg` has provided a [compilation guide](https://trac.ffmpeg.org/wiki/CentosCompilationGuide) for CentOS but for lazy people, it is too much of an effort. :)

Finally, I found a solution that worked perfectly so I thought of sharing it here. Here’s the step by step guide.

**Step 1:** SSH into your instance and become root

```
sudo su -
```

**Step 2:** Go to the `/usr/local/bin` directory

```
cd /usr/local/bin
```

**Step 3:** Inside the `/usr/local/bin` directory, create an `ffmpeg` directory and go inside it.

```
mkdir ffmpeg && cd ffmpeg
```

**Step 4:** Download a static build of `ffmpeg`

Go to the following link and download a static build relevant for your system.

[FFmpeg Static Builds](https://www.johnvansickle.com/ffmpeg/ "https://www.johnvansickle.com/ffmpeg/")

**Note:** To find the current build version from John’s site. Click on the “build info” link at the releases page and at the top of the `.txt` file, you will see the current version of the build. Replace the **ffmpeg-[X.X.X]-amd64-static** with the current build version.

To check if your system is 32-bit or 64-bit you can execute the following command:

```
uname -a
```

It will output something like:

```
Linux ip-172-31-1-100 4.9.62-21.56.amzn1.x86_64 #1 SMP Thu Nov 16 05:37:08 UTC 2017 x86_64 x86_64 x86_64 GNU/Linux
```

**Note:** That last `i386` indicates that it’s 32-bit; `x86_64` indicates 64-bit.

Also, by hit and trial, I found out that the processor is AMD and not ARM. Inside the `/usr/local/bin/ffmpeg` folder run this command to download the static binaries.

```
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
```

**Step 5:** Unzip the binaries

Use the following command to unzip the binaries.

```
tar -xf ffmpeg-release-amd64-static.tar.xz
```

This will create a folder named `ffmpeg-4.2.1-amd64-static` . Go inside this folder to check if `ffmpeg` has been installed successfully or not.

```
./ffmpeg -version
```

It should output something like:

```
ffmpeg version 4.2.1-static https://johnvansickle.com/ffmpeg/  

.
.
.
.
Hyper fast Audio and Video encoder
```

Now we will move to the outer folder.

```
cp -a /usr/local/bin/ffmpeg/ffmpeg-4.2.1-amd64-static/ . /usr/local/bin/ffmpeg/
```

**Step 6:** Create a symlink to use `ffmpeg` from any location

Next, go ahead and create a symlink so that `ffmpeg` can be run from any location.

```
ln -s /usr/local/bin/ffmpeg/ffmpeg /usr/bin/ffmpeg
```

Note: The first part `/usr/local/bin/ffmpeg/ffmpeg` is where the file is located after I untarred the file. The second part `/usr/bin/ffmpeg` is where we want the symlink to go

That’s it. You are done with the installation.

I found this solution on AWS forums and have shared it here so that it is easier to find. Here’s a link to the forum post.

[https://forums.aws.amazon.com/thread.jspa?messageID=332091](https://forums.aws.amazon.com/thread.jspa?messageID=332091)

If you are trying to install FFmpeg on a Raspberry Pi then check out this post for step by step instructions:

[How to install FFMPEG and FFServer on Raspberry Pi](/how-to-install-ffmpeg-and-ffserver-on-raspberry-pi-ed0eddf86f88 "https://medium.com/@maskaravivek/how-to-install-ffmpeg-and-ffserver-on-raspberry-pi-ed0eddf86f88")

You can buy me a coffee if this post really helped you learn something or fix a nagging issue!

* * *
Written on October 21, 2019 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/how-to-install-ffmpeg-on-ec2-running-amazon-linux-451e4a8e2694)
