---
title: "How to access files from Google Cloud Storage in Colab Notebooks"
author: "Vivek Maskara"
date: 2020-06-08T05:38:39.003Z
lastmod: 2021-10-29T21:25:44-07:00

description: ""

subtitle: ""

categories: [Google Cloud]

tags:
 - Google Cloud Storage
 - Colab Notebooks

image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2020-06-08_how-to-access-files-from-google-cloud-storage-in-colab-notebooks_0.jpeg"
 - "/post/img/2020-06-08_how-to-access-files-from-google-cloud-storage-in-colab-notebooks_1.png"
 - "/post/img/2020-06-08_how-to-access-files-from-google-cloud-storage-in-colab-notebooks_2.png"


aliases:
- "/how-to-access-files-from-google-cloud-storage-in-colab-notebooks-8edaf9e6c020"

---

![](/post/img/2020-06-08_how-to-access-files-from-google-cloud-storage-in-colab-notebooks_0.jpeg#layoutTextWidth)

Recently, while working with a large dataset, I wanted to use Google Cloud Storage in my Colab notebook. In this post, I will show how to access these files in Google Colab Notebook.

It turns out that even if your file is public, you can’t do a simple `curl` or `wget` to access the file stored on Google Cloud Storage.

Here’s how you can upload and download files on Google Cloud Storage.

#### Step 1: Authenticate using your Google Account

Firstly, you need to authenticate yourself in Colab. Once you run the code below, it will ask you to follow a link to login and enter an access token that you receive upon successful login.

```
from google.colab import auth

auth.authenticate_user()
```

#### Step 2: Install the GCloud SDK

We would be using the `gsutil` command to upload and download files. So we first need to install the GCloud SDK.

```
!curl https://sdk.cloud.google.com | bash
```

#### Step 3: Init the SDK

Next, init the SDK to configure the project settings.

```
!gcloud init
```

Once, you run the above command, it will ask you a few questions to configure the SDK.

![](/post/img/2020-06-08_how-to-access-files-from-google-cloud-storage-in-colab-notebooks_1.png#layoutTextWidth)

#### Step 4: Upload and Download files

Finally, you are all set to upload and download files using Google Cloud Storage.

**Download file from Cloud Storage to Google Colab**

```
!gsutil cp gs://maskaravivek-data/data_file.csv .
```

Upload file from Google Colab to Cloud

```
gsutil cp test.csv gs://maskaravivek-data/
```

That’s it. Here’s the Colab notebook for your reference:

[Google Colaboratory](https://colab.research.google.com/drive/1KyySbZO0hjCxQUs3F6pSdE2Ze1PC7OCJ?usp=sharing "https://colab.research.google.com/drive/1KyySbZO0hjCxQUs3F6pSdE2Ze1PC7OCJ?usp=sharing")

Using Google Cloud Storage might not be the ideal solution for you and most of the time mounting Google Drive should suffice.

You can buy me a coffee if this post really helped you learn something or fix a nagging issue!

* * *
Written on June 8, 2020 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/how-to-access-files-from-google-cloud-storage-in-colab-notebooks-8edaf9e6c020)
