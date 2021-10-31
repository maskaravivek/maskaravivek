---
title: "Handy Shell Commands for Beginners"
author: "Vivek Maskara"
date: 2019-05-10T19:14:38.443Z
lastmod: 2021-10-29T21:24:20-07:00

description: ""

subtitle: ""

categories: [Shell]

tags:
 - Shell

image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2019-05-10_handy-shell-commands-for-beginners_0.gif"


aliases:
- "/handy-shell-commands-for-beginners-295f47161040"

---

![](/post/img/2019-05-10_handy-shell-commands-for-beginners_0.gif#layoutTextWidth)

#### Check if a file exists

```
if [ ! -f "/home/pi/test.sh" ]; then
    echo "file exists"
fi
```

#### Check if a directory exists

```
if [ -d "/home/pi/test" ]; then
    echo "test dir exists"
fi
```

#### Install Apt-Get Package

```
# update pip
# check if package is installed
# install package if not installed
installPackage() {
    retval=0
    echo "Installing package $1"
    if [ $(dpkg-query -W -f='${Status}' $1 2>/dev/null | grep -c "ok installed") -eq 0 ];
    then
        echo "Package $1 is not installed"
        if echo "yes" | sudo apt-get install $1; then
            retval=1
        fi
    else
        echo "Package $1 is already installed"
        retval=1
    fi
    return "$retval"
}
```

#### Manually Set Date Time

```
sudo date -s '2019-05-10 09:07:00'
```

#### Setup SSH Keys

```
mkdir -p /home/pi/.ssh

sudo echo '' | ssh-keygen -t rsa -q -N '' -f /home/pi/.ssh/id_rsa
```

#### Turn Off Strict Host Checking For a Domain

```
echo -e "Host google.com\n\tStrictHostKeyChecking no\n" >> ~/.ssh/config
```

* * *
Written on May 10, 2019 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/handy-shell-commands-for-beginners-295f47161040)
