---
title: "How to Stream Live Feed From DSLR on Raspberry Pi using FFServer"
author: "Vivek Maskara"
date: 2020-10-17T07:44:28.998Z
lastmod: 2021-10-29T21:26:10-07:00

description: ""

subtitle: ""

categories: [Raspberry Pi]

tags:
 - Live Streaming
 - DSLR
 - Raspberry Pi
 - FFServer
 - Live Feed

image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2020-10-17_how-to-stream-live-feed-from-dslr-on-raspberry-pi-using-ffserver_0.jpeg"
 - "/post/img/2020-10-17_how-to-stream-live-feed-from-dslr-on-raspberry-pi-using-ffserver_1.png"


aliases:
- "/how-to-stream-live-feed-from-dslr-on-raspberry-pi-using-ffserver-839d8d864cce"

---

![](/post/img/2020-10-17_how-to-stream-live-feed-from-dslr-on-raspberry-pi-using-ffserver_0.jpeg#layoutTextWidth)

In this short post, I will walk you through the steps for streaming a live camera feed from a DSLR connected to the Raspberry Pi. We will be using [gphoto2](http://www.gphoto.org/) for interfacing with the camera, FFmpeg for encoding the video, and FFserver for hosting the feed on a local webserver.

### Prerequisites

#### Hardware

For this tutorial, I am assuming that you have a Raspberry Pi with Raspbian or Noob OS installed on it. You can get a [Raspberry Pi](https://amzn.to/3jJoeLC) from Amazon if you don’t already have one.

Also, I am assuming that you already have one of the supported cameras listed here:

[Projects :: libgphoto2 :: supported cameras](http://www.gphoto.org/proj/libgphoto2/support.php "http://www.gphoto.org/proj/libgphoto2/support.php")

I used the [Canon Rebel T7 camera](https://amzn.to/3jIBccL) which I got from Amazon.

#### Software

Install libghoto2 and ghoto2. Check out the setup instructions in this post for installing these libraries on a Rasberry Pi.

[How to Control and Capture Images from DSLR using Raspberry Pi](/how-to-control-and-capture-images-from-dslr-using-raspberry-pi-fdfa9d600ec1 "https://medium.com/@maskaravivek/how-to-control-and-capture-images-from-dslr-using-raspberry-pi-fdfa9d600ec1")

Install FFmpeg and FFserver. Check out this post for step-by-step instructions.

[How to install FFMPEG and FFServer on Raspberry Pi](/how-to-install-ffmpeg-and-ffserver-on-raspberry-pi-ed0eddf86f88 "https://medium.com/@maskaravivek/how-to-install-ffmpeg-and-ffserver-on-raspberry-pi-ed0eddf86f88")

### Setup FFServer

Create a file at `/etc/ffserver.conf` and the following contents in it.

```
Port 8090
BindAddress 0.0.0.0
MaxHTTPConnections 2000
MaxClients 1000
MaxBandwidth 1000
CustomLog -
NoDaemon

<Feed feed1.ffm>
        File /tmp/feed1.ffm
        FileMaxSize 200M
        ACL allow 127.0.0.1
</Feed>

<Stream test1.swf>
        Feed feed1.ffm
        Format mpeg

        AudioBitRate 32
        AudioChannels 1
        AudioSampleRate 44100

        VideoBitRate 64
        VideoBufferSize 40
        VideoFrameRate 20
        VideoSize 1056x704
        VideoGopSize 12

        NoAudio
</Stream>

<Stream stat.html>
        Format status
        ACL allow localhost
        ACL allow 192.168.0.0 192.168.255.255
</Stream>

<Redirect index.html>
        URL http://www.ffmpeg.org/
</Redirect>
```

Note: I am using the configs as mentioned in this [thread](http://gphoto-software.10949.n7.nabble.com/liveview-in-EOS-cameras-td4238.html). Checkout [FFserver docs](https://trac.ffmpeg.org/wiki/ffserver) for more configuration options.

Start the server:

```
ffserver -d -f /etc/ffserver.conf
```

### Publish the Stream

We would be using ghoto2 to capture live camera preview from the DSLR and will then be encoding it to `mjpg` format using FFmpeg. Finally, we will be serving this file to `feed1.ffm` as defined in the `ffserver.conf` file. This command achieves all of the above:

```
gphoto2 --capture-movie --stdout | ffmpeg -re -i pipe:0 -listen 1 -f swf http://localhost:8090/feed1.jpg
```

### View the Stream

Create a simple HTML page and embed the stream URL in it.

```
<html>
    <body>
        <object style="width: 600px; height: 600px;" data="http://192.168.0.4:8090/test1.swf"></object>
    </body>
</html>
```

Open the HTML file in the browser and you should be able to view the stream.

Note: You will need to enable Flash support in your browser. Also, keep in mind that Flash support might permanently be gone from the browsers by the end of the year.

That’s it for this post. I hope you will find this post useful. I found the above information in a comment on [this thread](http://gphoto-software.10949.n7.nabble.com/liveview-in-EOS-cameras-td4238.html). All credits to [vincent](http://gphoto-software.10949.n7.nabble.com/template/NamlServlet.jtp?macro=user_nodes&user=2089) for suggesting this solution.

You can buy me a coffee if this post really helped you learn something or fix a nagging issue!

* * *
Written on October 17, 2020 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/how-to-stream-live-feed-from-dslr-on-raspberry-pi-using-ffserver-839d8d864cce)
