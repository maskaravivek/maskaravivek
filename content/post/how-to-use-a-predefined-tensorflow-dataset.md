---
title: "How to use a pre-defined Tensorflow Dataset?"
author: "Vivek Maskara"
date: 2020-06-18T02:35:05.458Z
lastmod: 2021-10-29T21:26:00-07:00

description: ""

subtitle: ""

categories: [Deep Learning, TensorFlow]

tags:
 - Dataset
 - TensorFlow
 - Deep Learning
 - Image Net

image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2020-06-18_how-to-use-a-predefined-tensorflow-dataset_0.jpeg"
 - "/post/img/2020-06-18_how-to-use-a-predefined-tensorflow-dataset_1.png"


aliases:
- "/how-to-use-a-pre-defined-tensorflow-dataset-ce6923e6d7f2"

---

![](/post/img/2020-06-18_how-to-use-a-predefined-tensorflow-dataset_0.jpeg#layoutTextWidth)

Tensorflow 2.0 comes with a set of pre-defined ready to use datasets. It is quite easy to use and is often handy when you are just playing around with new models.

In this short post, I will show you how you can use a pre-defined Tensorflow Dataset.

### Prerequisite

Make sure that you have `tensorflow` and `tensorflow-datasets` installed.

```
pip install -q tensorflow-datasets tensorflow
```

### Using a Tensorflow dataset

In this example, we will use a small `imagenette` dataset.

[imagenette | TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/imagenette "https://www.tensorflow.org/datasets/catalog/imagenette")

You can visit this [link](https://www.tensorflow.org/datasets/catalog/overview) to get a complete list of available datasets.

#### **Load the dataset**

We will use the `tfds.builder` function to load the dataset.

```
import tensorflow_datasets as tfds

imagenette_builder = tfds.builder("imagenette/full-size")
imagenette_info = imagenette_builder.info
imagenette_builder.download_and_prepare()
datasets = imagenette.as_dataset(as_supervised=True)
```

**Note:**

- we are setting `as_supervised` as `true` so that we can perform some manipulations on the data.
- we are creating an `imagenette_info` object that contains the information about the dataset. It prints something like this:

![](/post/img/2020-06-18_how-to-use-a-predefined-tensorflow-dataset_1.png#layoutTextWidth)

#### Get split size

We can get the size of the train and validation set using the `imagenette_info` object.

```
train_examples = imagenette_info.splits['train'].num_examples
validation_examples = imagenette_info.splits['validation'].num_examples
```

This would be useful while defining the `steps_per_epoch` and `validation_steps` of the model.

#### Create batches

Next, we will create batches so that the data is easily trainable. On low RAM devices or for large datasets it is usually not possible to load the whole dataset in memory at once.

```
train, test = datasets['train'], datasets['validation']

train_batch = train.map(
    lambda image, label: (tf.image.resize(image, (448, 448)), label)).shuffle(100).batch(batch_size).repeat()

validation_batch = test.map(
    lambda image, label: (tf.image.resize(image, (448, 448)), label)
).shuffle(100).batch(batch_size).repeat()
```

Note: We are taking the train and validation splits and resizing all images to `448 x 448` . You can perform any other manipulation too using the `map` function. It is useful to resize or normalize the image or perform any other preprocessing step.

That’s it. You can now use this data for your model. Here’s the link to the Google Colab with the complete code.

[Google Colaboratory](https://colab.research.google.com/drive/1EtXNakSFJs6d5XvdPQepYlzf8Sa8bCi6?usp=sharing "https://colab.research.google.com/drive/1EtXNakSFJs6d5XvdPQepYlzf8Sa8bCi6?usp=sharing")

* * *
Written on June 18, 2020 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/how-to-use-a-pre-defined-tensorflow-dataset-ce6923e6d7f2)
