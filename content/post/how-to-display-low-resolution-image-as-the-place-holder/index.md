---
title: "How to Display Low Resolution Image as the Place Holder"
author: "Vivek Maskara"
date: 2019-09-21T20:31:06.385Z
lastmod: 2021-10-29T21:24:24-07:00

description: ""

subtitle: ""

categories: [Android]

tags:
 - Android
 - Fresco

image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2019-09-21_how-to-display-low-resolution-image-as-the-place-holder_0.jpeg"


aliases:
- "/how-to-display-low-resolution-image-as-the-place-holder-ab59d0f44d8a"

---

![](/post/img/2019-09-21_how-to-display-low-resolution-image-as-the-place-holder_0.jpeg#layoutTextWidth)

Fresco is a powerful system for displaying images in Android applications.

There are situations where the original image is quite big and would take considerable amount of time to load. In such scenarios, it is advisable to display a low resolution image or thumbnail image until the original image loads. [Fresco](https://github.com/facebook/fresco) makes it very easy to implement.

#### Using Fresco in your application

If you are building with Gradle, simply add the following line to the `dependencies` section of your `build.gradle` file:

```
implementation 'com.facebook.fresco:fresco:2.0.0'
```

For more information on adding Fresco to your app, check out their [repository](https://github.com/facebook/fresco).

#### Initialise Fresco in Your App

Fresco needs to be initialised. You should only do this 1 time, so placing the initialisation in your `Application` is a good idea.

```
public class MyApplication extends Application {
    @Override
    public void onCreate() {
        super.onCreate();
        Fresco.initialize(this);
    }
}
```

#### Add Drawee View for Fresco

Add the `SimpleDraweeView` to the layout:

```
<com.facebook.drawee.view.SimpleDraweeView
    android:id="@+id/my_image_view"
    android:layout_width="130dp"
    android:layout_height="130dp"
    fresco:placeholderImage="@drawable/my_drawable"
    />
```

#### Using a low-resolution image as the place holder

First, bind the `SimpleDraweeView` to your View.

```
@BindView(R.id.mediaDetailImage)    
SimpleDraweeView image;
```

Next, setup your image view using `newDraweeControllerBuilder`

```
private void setupImageView() {
    DraweeController controller = Fresco.newDraweeControllerBuilder()          .setLowResImageRequest(ImageRequest.fromUri(media.getThumbUrl()))           .setImageRequest(ImageRequest.fromUri(media.getImageUrl()))
            .setOldController(image.getController())
            .build();
    image.setController(controller);
}
```

Fresco lets you specify a low res and original image request.

- use `setLowResImageRequest` to set the temporary image. It need not be a static placeholder. It can be fetched and displayed from the server.
- use `setImageRequest` to set the original image.

You can read more about `ImageRequests`[here](https://frescolib.org/docs/image-requests.html). Also, there’s more documentation about [requesting multiple images](https://frescolib.org/docs/requesting-multiple-images.html) here.

* * *
Written on September 21, 2019 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/how-to-display-low-resolution-image-as-the-place-holder-ab59d0f44d8a)
