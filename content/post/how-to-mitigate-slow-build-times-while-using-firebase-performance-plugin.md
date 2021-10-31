---
title: "How to Mitigate Slow Build Times While Using Firebase Performance Plugin"
author: "Vivek Maskara"
date: 2018-06-14T04:36:31.618Z
lastmod: 2021-10-29T21:23:59-07:00

description: ""

subtitle: ""

categories: [Android]

tags:
 - Android
 - Firebase
 - Gradle


image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2018-06-14_how-to-mitigate-slow-build-times-while-using-firebase-performance-plugin_0.png"
 - "/post/img/2018-06-14_how-to-mitigate-slow-build-times-while-using-firebase-performance-plugin_1.png"


aliases:
- "/how-to-mitigate-slow-build-times-while-using-firebase-performance-plugin-c40e9422cf7a"

---

I recently added [Firebase Performance Plugin](https://firebase.google.com/docs/perf-mon/) to one of my projects and experienced a drastic increase in the build time of the app. The app initially used to build in under 20 seconds and with the addition of this plugin, it started taking around 5–6 minutes for a single build. This was quite annoying and I looked for what might be causing the increase in build times.

If you look closely you will find out that the following task is taking too long to build:

```
app:transformClassesWithFirebasePerformancePluginForDebug
```

There is a post-compilation phase when using Firebase Performance on Android that results in slow build times.

### How to Mitigate the Issue

The fix that applied only mitigates the issue by adding a parameter to build command so that this plugin can be disabled during development.

In `root/buildscript/dependencies` wrap the plugin `classpath` inside the following `if` condition.

```
if (!project.hasProperty("disable-performance-plugin")) {
    classpath('com.google.firebase:firebase-plugins:1.1.5') {
        exclude group: 'com.google.guava', module: 'guava-jdk5'
    }
}
```

Excluding `com.google.guava` depends on whether it is causing conflicts with your Guava dependencies as mentioned in the [getting started](https://firebase.google.com/docs/perf-mon/get-started-android) documentation.

Next, in `app/build.gradle` wrap the apply plugin code in the following condition.

```
if (!project.hasProperty("disable-performance-plugin")) { 
    apply plugin: 'com.google.firebase.firebase-perf' 
}
```

Now, simply build using the command line with the parameter for disabling the plugin.

```
./gradlew your-task -Pdisable-performance-plugin
```

If you use Android Studio for building the project then you add the same config in **Compiler** settings of Studio. Set the command line options to:

```
-Pdisable-performance-plugin
```

![](/post/img/2018-06-14_how-to-mitigate-slow-build-times-while-using-firebase-performance-plugin_0.png#layoutTextWidth)

That’s it. Adding this parameter should make your life easier! :)

Credits to [Ivan Kravchenko](https://medium.com/@kravchenko.i.vl) for the [original answer](https://stackoverflow.com/questions/47380524/firebase-performance-plugin-causing-slow-build-time/47994657#47994657) on Stack Overflow.

You can buy me a coffee if this post really helped you learn something or fix a nagging issue!

* * *
Written on June 14, 2018 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/how-to-mitigate-slow-build-times-while-using-firebase-performance-plugin-c40e9422cf7a)
