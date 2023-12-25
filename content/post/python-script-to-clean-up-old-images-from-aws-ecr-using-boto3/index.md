---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Python Script to Clean up Old Images From AWS ECR using Boto3"
subtitle: ""
summary: ""
authors: [admin]
tags: [AWS, Python]
categories: [AWS]
date: 2023-12-24T23:16:01-07:00
lastmod: 2023-12-24T23:16:01-07:00
featured: true
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

For one of my projects, I automated [building and pushing AWS ECR images using Github Actions](https://medium.com/geekculture/github-actions-pipeline-to-push-docker-images-to-amazon-ecr-4bca6ec864bd), but soon realized that my AWS ECR repo had a pile-up of unused images. In my workflows, I use the most recently image and other images in the repository present a minor opportunity for optimization. Since, AWS ECR [charges](https://aws.amazon.com/ecr/pricing/) you based on the storage size, cleaning up old images will save me a few bucks.

In this short post, I will share a Python script that is quite handy for cleaning up old/unused images from [AWS ECR](https://aws.amazon.com/ecr/).

## Python Boto3 Script

First create a `ecr-cleanup` directory to hold the `requirements.txt` and `main.py` files. We will use [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) library for fetching image details in a repo and deleting them. So first create a `requirements.txt` file with the following contents:

```txt
boto3==1.34.7
```

You can install the dependency using the following command:

```bash
pip install -r requirements.txt
```

Next, create a `main.py` file and add the following code snippet to it:

```python
import boto3

def fetch_all_images(repository_name):
    ecr_client = boto3.client('ecr')
    images = []
    next_token = None

    while True:
        if next_token:
            response = ecr_client.list_images(repositoryName=repository_name, nextToken=next_token)
        else:
            response = ecr_client.list_images(repositoryName=repository_name)

        images.extend(response['imageIds'])
        if 'nextToken' in response:
            next_token = response['nextToken']
        else:
            break

    return images

def delete_images(repository_name, image_ids):
    if len(image_ids) == 0:
        print("No images to delete.")
        return
    ecr_client = boto3.client('ecr')
    response = ecr_client.batch_delete_image(repositoryName=repository_name, imageIds=image_ids)
    deleted_images = response['imageIds']
    print(f"Deleted {len(deleted_images)} images.")

def sort_images_by_push_date(images):
    ecr_client = boto3.client('ecr')
    sorted_images = sorted(images, key=lambda x: ecr_client.describe_images(repositoryName=repository_name, imageIds=[x])['imageDetails'][0]['imagePushedAt'], reverse=True)
    return sorted_images

def delete_all_except_recent(repository_name):
    images = fetch_all_images(repository_name)
    sorted_images = sort_images_by_push_date(images)
    images_to_delete = sorted_images[1:]  # Exclude the most recent image
    delete_images(repository_name, images_to_delete)

# Usage
import os
repository_name = os.environ['REPO_NAME']
delete_all_except_recent(repository_name)
```

The script performs the following operations:

* It reads the `REPO_NAME` environment variable which corresponds to the AWS ECR repo name.
* Next, it fetches all images from the repo using the `fetch_all_images`.
* The `sort_images_by_push_date` returns a sorted list of images based on their push date.
* Finally, `delete_images` is invoked to delete all images except the last one.

You can run the script using the following command:

```bash
export REPO_NAME=my-repo-name && python main.py
```

Note: The script assumes that you have [configured AWS credentials](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html) on your shell before executing it.

That's it for this post. I hope you find this post useful!
