---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Handling corrupted data in Pytorch Dataloader"
subtitle: ""
summary: ""
authors: []
tags: [Pytorch, Deep Learning]
categories: [Deep Learning, Pytorch]
date: 2021-10-02T02:24:09-07:00
lastmod: 2021-10-02T02:24:09-07:00
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: true

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

Recently, while working on a video dataset, I noticed that some of the videos contained a few corrupted frames. While running the training, my dataloader used to return an incorrect shaped tensor since it was not able to read the corrupted frame. In this post, I will walk you through the process of utlilising Pytorch's `collate_fn` for overcoming this issue. I came across this solution in a [Github comment](https://github.com/pytorch/pytorch/issues/1137#issuecomment-618286571) posted by [Tsveti Iko](https://github.com/tsvetiko). 

First and foremost, start returning `None` from the dataset's `__getitem__` function for corrupted item. For eg. for my video dataset, I started checking the shape of the tensor and returned `None` if it didn't match the expected shape. The expected shape was (30, 3, 224, 224). 

```
from torch.utils.data import Dataset
import torch
import torchvision

class VideoDataset(Dataset):

    ...
    
    def __getitem__(self, idx):
        labels = labels[idx]
        labels = torch.tensor(labels)
        
        vid, audio, dict = torchvision.io.read_video(filename=video_list[idx])

        if vid.shape[0]!= 30 or vid.shape[1]!=3 or vid.shape[2]!=224 or vid.shape[3]!=224:
            return None
        
        return vid, labels
```

Next, define a collate function that filters the None records. 

```
import torch

def collate_fn(self, batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
```

Finally, use the `collate_fn` while defining the dataloader. 

```
from torch.utils.data import DataLoader
import os

dataset = VideoDataset()

dataloader = DataLoader(dataset, 
    batch_size=4, 
    shuffle=True, 
    num_workers=os.cpu_count() - 1, 
    pin_memory=True,
    collate_fn=collate_fn)
```