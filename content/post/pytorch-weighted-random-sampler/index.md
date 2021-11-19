---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Using Weighted Random Sampler in PyTorch"
subtitle: ""
summary: ""
authors: [admin]
tags: [PyTorch, Dataloader]
categories: [Deep Learning, PyTorch]
date: 2021-11-19T00:56:16-07:00
lastmod: 2021-11-19T00:56:16-07:00
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

Sometimes there are scenarios where you have way lesser number of samples for some of the classes where as other classes have lots of samples. In such a scenario, you don't want a training batch to be contain samples just from a few of the classes with lots of samples. Ideally, a training batch should contain represent a good spread of the dataset. In PyTorch this can be achieved using a [weighted random sampler](https://pytorch.org/docs/stable/data.html#:~:text=Generator%20used%20in%20sampling.-,CLASS,-torch.utils.data.WeightedRandomSampler). 

In this short post, I will walk you through the process of creating a random weighted sampler in PyTorch. 

To start off, lets assume you have a dataset with images grouped in folders based on their class. We can use a `ImageFolder` to create a dataset from it. 

```
from torchvision import datasets

dataset = datasets.ImageFolder(root=data_dir, transform=image_transforms)
```

Also, lets split this dataset into training and validation sets, 

```
dataset_size = dataset.__len__()
train_count = int(dataset_size * 0.7)
val_count = dataset_size - train_count
train_dataset, valid_dataset = random_split(dataset, [train_count, val_count])
```

Note: I am taking a `ImageFolder` and training/validation splits just to emulate a real world example. You can work with any pytorch dataset. 

We will be using a weighted random sampler just for the training set. For validation set, we don't care about balancing a batch. Now that we have the `train_dataset`, you need to define the weights for each class which would be inversely proportional to the number of samples for each class. 

First, lets find the number of samples for each class. 

```
import numpy as np 

y_train_indices = train_dataset.indices

y_train = [dataset.targets[i] for i in y_train_indices]

class_sample_count = np.array(
    [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
```

Next, we need to find the weights for each class. 

```
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in y_train])
samples_weight = torch.from_numpy(samples_weight)
```

Now, that we have the weights for each of the classes, we can define a sampler. 

```
sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
```

Finally, we can use the sampler, while defining the `Dataloader`. 

```
train_dataloader = DataLoader(train_dataset, batch_size=4, sampler=sampler)
```

Thats it for this post. I hope you found this post useful. 