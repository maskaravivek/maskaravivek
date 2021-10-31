---
title: "How to SSH into an EC2 instance using Boto3"
author: "Vivek Maskara"
date: 2020-04-26T00:48:26.511Z
lastmod: 2021-10-29T21:25:18-07:00

description: ""

subtitle: ""

categories: [AWS]

tags:
 - SSH
 - Boto3
 - AWS
 - EC2

image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2020-04-26_how-to-ssh-into-an-ec2-instance-using-boto3_0.png"
 - "/post/img/2020-04-26_how-to-ssh-into-an-ec2-instance-using-boto3_1.png"


aliases:
- "/how-to-ssh-into-an-ec2-instance-using-boto3-a138a4345a91"

---

![](/post/img/2020-04-26_how-to-ssh-into-an-ec2-instance-using-boto3_0.png#layoutTextWidth)

The simplest way to programmatically SSH into an EC2 instance is to using the [Paramiko](https://github.com/paramiko/paramiko) package. In this post, I will show you how to use this package to SSH with multiple retries and then execute your commands on it.

### Preresiquite

#### Install Boto3

We will be using python for our code so make sure that you have `boto3` library installed.

```
pip install boto3
```

Next, you will need to configure the credentials. Please follow the [docs](https://pypi.org/project/boto3/) for the configuration steps.

### SSH into EC2 using Boto3

#### Get your instances

Firstly, we will use `boto3` to get the instance that we want to SSH into. The following code snippets give us the ip_address of the EC2 instance.

```
import boto3

# get your instance ID from AWS dashboard

ec2 = boto3.resource('ec2', region_name='us-east-1')
instance = ec2.Instance(id=instance_id)
instance.wait_until_running()
current_instance = list(ec2.instances.filter(InstanceIds=[instance_id]))
ip_address = current_instance[0].public_ip_address
```

#### Install Paramiko

Install Paramiko using Pip. If you are having trouble installing the package, refer to their [installation guide](http://www.paramiko.org/installing.html).

```
pip install paramiko
```

#### Using Paramiko

First, create an instance of Paramiko SSH client:

```
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
```

#### **SSH with retries**

Use the following method to SSH with retries. While using the library I noticed that out if you try SSHing into the instance just after the instance is started, it sometimes fails. So I have added a simple recursive retrial logic that tries connecting thrice after an interval of 5 seconds each.

```
def ssh_connect_with_retry(ssh, ip_address, retries):
    if retries > 3:
        return False
    privkey = paramiko.RSAKey.from_private_key_file(
        './config/image_rec_auth.pem')
    interval = 5
    try:
        retries += 1
        print('SSH into the instance: {}'.format(ip_address))
        ssh.connect(hostname=ip_address,
                    username='ubuntu', pkey=privkey)
        return True
    except Exception as e:
        print(e)
        time.sleep(interval)
        print('Retrying SSH connection to {}'.format(ip_address))
        ssh_connect_with_retry(ssh, ip_address, retries)
```

Call this method to SSH into the instance:

```
ssh_connect_with_retry(ssh, ip_address, 0)
```

**Note:** You can also get the public IP of your EC2 instance from the AWS dashboard.

#### Run your commands

Once you have connected to your instance, you can run your commands and get back the std output:

```
stdin, stdout, stderr = ssh.exec_command("echo 'Hello World!'")
print('stdout:', stdout.read())
print('stderr:', stderr.read())
```

Note: If you are trying to run multiple commands at once, you can probably read all the commands from a text file and to make it easily manageable.

#### Putting it all together

Here’s the complete code snippet:

```
import boto3

def ssh_connect_with_retry(ssh, ip_address, retries):
    if retries > 3:
        return False
    privkey = paramiko.RSAKey.from_private_key_file(
        './config/image_rec_auth.pem')
    interval = 5
    try:
        retries += 1
        print('SSH into the instance: {}'.format(ip_address))
        ssh.connect(hostname=ip_address,
                    username='ubuntu', pkey=privkey)
        return True
    except Exception as e:
        print(e)
        time.sleep(interval)
        print('Retrying SSH connection to {}'.format(ip_address))
        ssh_connect_with_retry(ssh, ip_address, retries)

# get your instance ID from AWS dashboard

# get instance
ec2 = boto3.resource('ec2', region_name='us-east-1')
instance = ec2.Instance(id=instance_id)
instance.wait_until_running()
current_instance = list(ec2.instances.filter(InstanceIds=[instance_id]))
ip_address = current_instance[0].public_ip_address

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

ssh_connect_with_retry(ssh, ip_address, 0)

stdin, stdout, stderr = ssh.exec_command(commands)
print('stdout:', stdout.read())
print('stderr:', stderr.read())
```

You can buy me a coffee if this post really helped you learn something or fix a nagging issue!

* * *
Written on April 26, 2020 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/how-to-ssh-into-an-ec2-instance-using-boto3-a138a4345a91)
