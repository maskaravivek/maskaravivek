---
title: "Write your First Neural Network Using TensorFlow"
author: "Vivek Maskara"
date: 2020-03-01T02:26:04.456Z
lastmod: 2021-10-29T21:24:46-07:00

description: ""

subtitle: ""

categories: [Deep Learning]

tags:
 - Neural Networks
 - TensorFlow

image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2020-03-01_write-your-first-neural-network-using-tensorflow_0.jpeg"


aliases:
- "/write-your-first-neural-network-using-tensorflow-6cfc67caff1b"

---

![](/post/img/2020-03-01_write-your-first-neural-network-using-tensorflow_0.jpeg#layoutTextWidth)

### Introduction

Neural Networks are a specific set of machine learning algorithms that are inspired by biological neural networks and they have revolutionized machine learning. In simple terms, they are general function approximations, which can be applied to almost any machine learning problem about learning a complex mapping from the input to the output space.

Neural networks can be quite complex but in this post, I will explain how to write your first neural network that tries to learn this linear relationship between x and y.

```
y = 2 * x - 1
```

The above equation is a simple linear equation and we don’t necessarily need a neural network to learn this relationship. But let us take this example to understand how you can write your first neural network using Python, Numpy and Tensorflow.

[NumPy](https://numpy.org/) is the fundamental package for scientific computing with Python.

[TensorFlow](https://www.tensorflow.org/) is an open-source platform for machine learning that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML-powered applications.

### Show me the Code

Let us start by import the required dependencies. We will be importing `tensorflow`, `numpy` and `keras`

```
import tensorflow as tf
import numpy as np
from tensorflow import keras
```

Create a neural network with 1 layer. The layer has 1 neuron, and the input shape to it is just 1 value.

```
model = tf.keras.Sequential([keras.layers.Dense(units =1, input_shape=[1])])
```

Compile the model using an optimizer and loss function.

- The LOSS function measures the guessed answers against the known correct answers and measures how well or how badly it did. For this particular case let us use a simple loss function ie. `mean_squared_error`
- The model uses an OPTIMIZER function to make another guess. Based on how the loss function went, it will try to minimize the loss. Let us use a stochastic gradient ascent as our optimizer.

```
model.compile(optimizer='sgd', loss='mean_squared_error')
```

Now that we have compiled our model, we can print the model summary and see the layers. Use the following statement to print the summary.

```
print(model.summary())
```

Here’s how the summary output looks. Notice that our model has just 1 layer that accepts 2 params and outputs 1 value.

```
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_2 (Dense)              (None, 1)                 2         
=================================================================
Total params: 2
Trainable params: 2
Non-trainable params: 0
_________________________________________________________________
```

Now let us fit the model on some training data. As the relationship that we are trying to learn is quite simple, its sufficient to train the model on just a handful of examples.

We will call `model.fit()` 500 times ie. use `epochs` as 500 to learn the model. It just means that the model will try to optimize its weights for 500 iterations.

```
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
model.fit(xs, ys, epochs = 500)
```

On calling `model.fit()` we can see the output of the training. Notice that the loss keeps decreasing as the number of epochs increases.

```
Train on 6 samples
Epoch 1/10
6/6 [==============================] - 0s 264us/sample - loss: 1.6101e-09
Epoch 2/10
6/6 [==============================] - 0s 528us/sample - loss: 1.5790e-09
Epoch 3/10
6/6 [==============================] - 0s 327us/sample - loss: 1.5479e-09
.
.
.
Epoch 270/500
6/6 [==============================] - 0s 385us/sample - loss: 0.0062
.
.
.
.
Epoch 498/500
6/6 [==============================] - 0s 225us/sample - loss: 5.4479e-05
Epoch 499/500
6/6 [==============================] - 0s 298us/sample - loss: 5.3360e-05
Epoch 500/500
6/6 [==============================] - 0s 768us/sample - loss: 5.2264e-05
```

Finally, let us perform predictions using the model. Use the following statement to check the output for input x = 10.0.

```
print(model.predict([10.0]))
```

The output for 500 epochs would be:

```
[[18.978909]]
```

I know you might have been expecting the output to be 19 but our neural network isn’t very sure. It has seen just a handful of training samples and the approximation function outputs a value that is very close to 19.0.

Anyways, the above example demonstrates how you can write a simple neural network, train it and use it for predictions.

Originally published at [http://maskaravivek.com](http://maskaravivek.com/2020/02/29/write-your-first-neural-network-using-tensorflow/) on February 29, 2020.

* * *
Written on March 1, 2020 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/write-your-first-neural-network-using-tensorflow-6cfc67caff1b)
