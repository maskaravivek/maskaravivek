---
title: "Using Custom Building Blocks in TensorFlow 2.0"
author: "Vivek Maskara"
date: 2020-06-22T06:53:14.082Z
lastmod: 2021-10-29T21:26:04-07:00

description: ""

subtitle: ""

categories: [Deep Learning, TensorFlow]

tags:
 - TensorFlow
 - Keras
 - Neural Networks
 - Deep Learning


image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2020-06-22_using-custom-building-blocks-in-tensorflow-2.0_0.jpeg"


aliases:
- "/using-custom-building-blocks-in-tensorflow-2-0-550b88eb7aa2"

---

#### Custom Data Generator, Layer, Loss function and Learning Rate Scheduler

![](/post/img/2020-06-22_using-custom-building-blocks-in-tensorflow-2.0_0.jpeg#layoutTextWidth)

In this post, I will demonstrate how you can use custom building blocks for your deep learning model. Specifically, we will see how to use custom data generators, custom Keras layer, custom loss function, and a custom learning rate scheduler.

### Custom Data Generator

Tensorflow provides `tf.keras.preprocessing.image.ImageDataGenerator` ([link](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)) which is apt for most of the use cases but in some cases you might want to use a custom data generator. You can implement the `keras.utils.Sequence` interface to define a custom generator for your problem statement.

```
from tensorflow import keras

class My_Custom_Generator(keras.utils.Sequence):
  
  def __init__(self, images, labels, batch_size):
    self.images = images
    self.labels = labels
    self.batch_size = batch_size
    
    
  def __len__(self) :
    return (np.ceil(len(self.images) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx) :
    batch_x = self.images[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

    train_image = []
    train_label = []

    for i in range(0, len(batch_x)):
      img_path = batch_x[i]
      label = batch_y[i]
      # read method takes image path and label and returns corresponding matrices
      image, label_matrix = read(img_path, label)
      train_image.append(image)
      train_label.append(label_matrix)
    return np.array(train_image), np.array(train_label)
```

**Note:**The `__getitem__` function returns a batch of images and labels.

Once, you define the generator, you can create its instances for training and validation sets.

You can check the shape of the generator by manually calling the `__getitem__` method.

Finally, pass the generators in the `model.fit` method.

Here’s the link to the [docs](https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence) if you want to explore it in detail.

### Custom Keras Layer

There might be scenarios where the inbuilt layers provided by Tensorflow might not be sufficient for your needs. You can extend the `tf.keras.layers.Layer` to implement your own layer. In the code snippet below I define a layer to reshape the output. I used this layer while [implementing YOLOV1 from scratch](https://www.maskaravivek.com/post/yolov1/).

**Note:**

- You need to define a `call` method that takes in the input from the previous layer and returns the output.
- Also, you would need to define `get_config` and call `config.update` in it to save the inputs to this layer. If you skip this, TensorFlow won’t be able to save the weights for the network.

Once you define a layer, you can use it like any other layer in your model.

```
model = Sequential().
.
.
model.add(Dense(1470, activation='linear'))
model.add(Yolo_Reshape(target_shape=(7,7,30)))
```

Here’s the link to the [docs](https://www.tensorflow.org/guide/keras/custom_layers_and_models) if you want to explore it in more detail.

### Custom Loss Function

Tensorflow comes with numerous predefined loss functions but there might be cases where you need to define your loss function. It is actually quite simple to define and use a custom loss function.

**Note:** The loss function takes in the `y_actual` and the `y_pred` tensors.

Once you define a loss function, you can use it like any other loss function.

```
from tensorflow import keras

model.compile(loss=custom_loss ,optimizer='adam')
```

Here’s the link to the [docs](https://www.tensorflow.org/tutorials/customization/custom_training#define_a_loss_function) if you want to explore it in more detail.

### Custom Learning Rate Scheduler

Some neural networks use variable learning rates for different epochs. If it doesn’t fit into one of the predefined learning rate schedulers, you can define your own. You need to extend the `keras.callbacks.Callback` class to define a custom scheduler.

`CustomLearningRateScheduler` takes a `schedule` when initialized. On the start of every epoch `on_epoch_begin` is triggered and there we set the value for the learning rate based on the schedule.

[Tensorflow callbacks](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback) have other methods like epoch end, batch begin, batch end, etc. and it can be used for many other purposes too.

Next, we need to define a schedule and a function that uses that schedule to retrieve the learning rate for a particular epoch.

Finally, we can use the scheduler in the `model.fit` function.

That’s it for this post. I used all these custom building blocks while trying to implement YOLOV1 from scratch. You can take a look at this post to see the full implementation:

[Implementing YOLOV1 from scratch using Keras Tensorflow | Vivek Maskara](https://www.maskaravivek.com/post/yolov1/ "https://www.maskaravivek.com/post/yolov1/")

* * *
Written on June 22, 2020 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/using-custom-building-blocks-in-tensorflow-2-0-550b88eb7aa2)
