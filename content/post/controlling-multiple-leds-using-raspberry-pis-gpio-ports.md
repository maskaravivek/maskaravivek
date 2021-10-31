---
title: "Controlling multiple LEDs using Raspberry Pi’s GPIO ports"
author: "Vivek Maskara"
date: 2020-05-17T22:35:31.505Z
lastmod: 2021-10-29T21:25:30-07:00

description: ""

subtitle: ""

categories: [Raspberry Pi]

tags:
 - IoT
 - Raspberry Pi
 - LEDs

image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2020-05-17_controlling-multiple-leds-using-raspberry-pis-gpio-ports_0.png"
 - "/post/img/2020-05-17_controlling-multiple-leds-using-raspberry-pis-gpio-ports_1.png"
 - "/post/img/2020-05-17_controlling-multiple-leds-using-raspberry-pis-gpio-ports_2.png"
 - "/post/img/2020-05-17_controlling-multiple-leds-using-raspberry-pis-gpio-ports_3.jpeg"
 - "/post/img/2020-05-17_controlling-multiple-leds-using-raspberry-pis-gpio-ports_4.jpeg"
 - "/post/img/2020-05-17_controlling-multiple-leds-using-raspberry-pis-gpio-ports_5.png"


aliases:
- "/controlling-multiple-leds-using-raspberry-pis-gpio-ports-c78173c102d3"

---

In this post, I will explain how you can control multiple LED lights using a Raspberry Pi. The whole thing will hardly take 15–20 minutes if you have all the pieces of equipment.

### Materials needed

The easiest way to get all the additional materials(apart from Raspberry Pi) is to get a starter kit instead of buying everything individually. I bought all the items from Amazon but you can find it anywhere.

- [Raspberry Pi 3](https://www.amazon.com/gp/product/B01C6EQNNK/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=B01C6EQNNK&linkCode=as2&tag=maskaravivek-20&linkId=7a85a611f3e58e5c77601b9a9516cdb7)
- [Breadboard, Resistors, LEDs and jumper wires](https://www.amazon.com/gp/product/B01ERP6WL4/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=B01ERP6WL4&linkCode=as2&tag=maskaravivek-20&linkId=1d3692be1aed1205a0c5d98c1f313377)

#### What is a GPIO Board?

According to Wikipedia, a [GPIO board](https://en.wikipedia.org/wiki/General-purpose_input/output) is defined as:

> A general-purpose input/output (GPIO) is an uncommitted digital signal pin on an integrated circuit or electronic circuit board whose behavior — including whether it acts as input or output — is controllable by the user at run time.

![](/post/img/2020-05-17_controlling-multiple-leds-using-raspberry-pis-gpio-ports_0.png#layoutTextWidth)

The Raspberry Pi has 2 rows of GPIO pins which can be used to connect LEDs or can be used to send signals to other devices. These pins facilitate the communication of Raspberry Pi with the outside world and let it control electronics and other devices.

The Raspberry Pi 3 has 26 GPIO pins, the rest of the pins are power, ground, or “other”.

You don’t need to understand all the pins of the GPIO board as we would be using just a few of the pins.

#### What is Breadboard?

A breadboard is a solderless board that makes it easy to prototype the electronic circuits. Its a plastic board with a grid of holes that are internally connected in specific ways using metal strips. As the breadboard does not need soldering, it makes it very easy to experiment and try out different combinations.

![](/post/img/2020-05-17_controlling-multiple-leds-using-raspberry-pis-gpio-ports_1.png#layoutTextWidth)

It has three different sections:

- The red power bus is used to connect power to the breadboard. You can connect power to any of the tie-points in the column.
- The blue ground bus which is used to connect ground to the breadboard. You can connect ground to any of the tie-points in the column.
- Rows of tie points from a to j. The left side tie-points are connected (A-B-C-D-E), and the right side tie-points are connected (F-G-H-I-J).

In our example, we would be using 2 LED lights but you can easily extend it to use 3–4 LED lights. In the next section, we will go ahead and build our own circuit using the breadboard, GPIO pins, LEDs, resistors, and jumper wires.

### Building the circuit

In our example, we would be using 3LED lights but you can easily extend it to use more LED lights. Build a circuit using the breadboard, GPIO pins, LEDs, resistors, and jumper wires as illustrated in the diagram below.

![](/post/img/2020-05-17_controlling-multiple-leds-using-raspberry-pis-gpio-ports_2.png#layoutTextWidth)

#### Single LED

After connecting the LEDs, our circuit looks something like this for a single LED setup.

![](/post/img/2020-05-17_controlling-multiple-leds-using-raspberry-pis-gpio-ports_3.jpeg#layoutTextWidth)

**Note**

- The circuit shown in Fig 3. instructs to use **RPi Pin 6 (GND)** to connect the black jumper wire. But you can use any GND pin if your Pin 6 is already in use. In Fig. 4 the jumper wire is connected to some other GND as Pin 6 was not free.
- We are connecting the red jumper wire to GPIO Pin 17.

#### Multiple LEDs

And this is how the circuit looks for Multiple LED setup.

![](/post/img/2020-05-17_controlling-multiple-leds-using-raspberry-pis-gpio-ports_4.jpeg#layoutTextWidth)

Note:

- We are using GPIO Pin 17, 18, and 22 for connecting the jumper wires with the Raspberry Pi. These ports numbers would be used later in the python code.

### Python Code

Now, that we have the circuits ready, we will go ahead and write simple python programs to control these LEDs.

#### Install the package

You might need to install the following [package](https://pypi.org/project/RPi.GPIO/) on your Raspberry Pi if it is not already installed.

```
pip install RPi.GPIO
```

This package will make it easy to communicate with Raspberry Pi’s GPIO pins.

#### Code for Single LED

The following code snippet can be used to control a single LED.

```
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(17,GPIO.OUT)
print "LED on"
GPIO.output(17,GPIO.HIGH)
time.sleep(10)
print "LED off"
GPIO.output(17,GPIO.LOW)
```

Note that we are passing `HIGH` and `LOW` voltage to the GPIO pin number **17** to control our LED light. Run the code to see if it works.

```
python single_led.py
```

#### Code for Multiple LEDs

The following code snippet can be used to control multiple LEDs.

```
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

GPIO.setup(17, GPIO.OUT)
GPIO.setup(18, GPIO.OUT)
GPIO.setup(22, GPIO.OUT)

GPIO.output(17, True)
time.sleep(3)
GPIO.output(17, False)
time.sleep(1)
GPIO.output(18, True)
time.sleep(3)
GPIO.output(18, False)
time.sleep(1)
GPIO.output(22, True)
time.sleep(3)
GPIO.output(22, False)
time.sleep(1)
```

Note that we are passing `HIGH` and `LOW` voltage to the GPIO pin number **17, 18, and 22**to control the 3 LEDs respectively. You can run the code and see if it works.

```
python multiple_led.py
```

You can buy me a coffee if this post really helped you learn something or fix a nagging issue!

* * *
Written on May 17, 2020 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/controlling-multiple-leds-using-raspberry-pis-gpio-ports-c78173c102d3)
