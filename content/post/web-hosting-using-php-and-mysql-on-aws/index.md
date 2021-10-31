---
title: "Web Hosting using PHP and MySQL on AWS"
author: "Vivek Maskara"
date: 2017-12-03T13:55:40.610Z
lastmod: 2021-10-29T21:23:18-07:00

description: ""

subtitle: ""

tags: [AWS, Web Hosting, PHP, MySQL, EC2]
categories: [AWS]

image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_0.png"
 - "/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_1.png"
 - "/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_2.png"
 - "/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_3.png"
 - "/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_4.png"
 - "/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_5.png"
 - "/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_6.png"
 - "/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_7.png"
 - "/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_8.png"
 - "/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_9.png"
 - "/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_10.png"
 - "/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_11.png"
 - "/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_12.png"
 - "/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_13.png"
 - "/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_14.png"
 - "/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_15.png"
 - "/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_16.png"
 - "/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_17.png"
 - "/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_18.png"
 - "/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_19.png"
 - "/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_20.png"
 - "/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_21.png"
 - "/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_22.png"
 - "/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_23.png"
 - "/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_24.png"


aliases:
- "/web-hosting-using-php-and-mysql-on-aws-95bd5df0bd75"

---

![](/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_0.png#layoutTextWidth)

It’s was a Sunday and I was bored. Then I came across a [custom URL shortener](https://github.com/YOURLS/YOURLS) service, **YOURLS** that I had previously used in my college days. It was very simple to set it up with my GoDaddy shared web hosting. I no longer have a web hosting package on GoDaddy so I decided to try out AWS cloud instead.

AWS is cheap and easy to get started. Our goal is to get our own custom URL shortener service up and running on [http://aloudapp.in](http://aloudapp.in).

Before we deep dive into the details of setting up everything, here are our broader goals.

- Setup an EC2 instance for cloud hosting
- Setup a hosting zone and configure DNS
- Setup a load balancer for the EC2 instance.
- Install a LAMP web server on the EC2 instance.
- Configure YOURLS on the web server.

This article will cover everything up to setting up the LAMP web server.

### Create an EC2 instance

An EC2 instance is a virtual server in Amazon’s Elastic Compute Cloud (EC2) for running applications on the Amazon Web Services (AWS) infrastructure.

#### Launch an EC2 instance

Click on **Launch Instance**to create a new instance.

![](/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_1.png#layoutTextWidth)

#### Run a Linux machine

Choose **Amazon Linux**from the listed options for machine images.

![](/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_2.png#layoutTextWidth)

#### Choose an Instance Type

We will go for a `t2.micro` free instance for the purpose of this example.

![](/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_3.png#layoutTextWidth)

#### Review your EC2 instance

All done! Review your settings.

![](/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_4.png#layoutTextWidth)

#### Create Key Pair

Create a public-private key pair if you desire. This is an optional step but we recommend that you create a key pair and download it. We will be using it later on in the article to SSH into our EC2 instance.

#### View Launch Status

Good Going! Your instance is up and running.

![](/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_5.png#layoutTextWidth)

#### Edit Security Group

Edit security group settings to allow SSH inbound traffic from your IP address.

![](/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_6.png#layoutTextWidth)

**Add a custom rule**

![](/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_7.png#layoutTextWidth)

Take a look at this post if you want to SSH into your EC2 instance programmatically.

[How to SSH into an EC2 instance using Boto3](/how-to-ssh-into-an-ec2-instance-using-boto3-a138a4345a91 "https://medium.com/@maskaravivek/how-to-ssh-into-an-ec2-instance-using-boto3-a138a4345a91")

### Domain Mapping

Next, you need to create a **Hosting Zone** where you would configure your DNS settings for the domain name. Here are the steps.

#### Create a Hosting zone

Click on **Create Hosting Zone** to get started.

![](/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_8.png#layoutTextWidth)

#### Setup Your Domain

Fill in your **Domain Name** to create a new hosted zone.

![](/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_9.png#layoutTextWidth)

#### Edit Namespace for your Domain

Once you have created a Hosted zone, you will be able to see associated namespace records for it. Edit these **NS**records for your domain. I purchased the domain from GoDaddy, so I had to go and edit these records in DNS settings for the domain on GoDaddy’s admin panel.

![](/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_10.png#layoutTextWidth)

#### Add a A Record

Coming back to your **Hosted Zone**, create a new **A**record. Leave the **Name**field empty if you don’t want to configure just for a particular subdomain.

![](/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_11.png#layoutTextWidth)

That’s it. Your **Hosted Zone** is now set up. You can also set up a subdomain in 4 easy steps by following this step by step guide.

[How to Create a Subdomain in Amazon Route 53?](/how-to-create-a-subdomain-in-amazon-route-53-f14a995c4295 "https://medium.com/@maskaravivek/how-to-create-a-subdomain-in-amazon-route-53-f14a995c4295")

### Create a Load Balancer

Next, you need to create a **Load Balancer** for your EC2 instance. As described by [Wikipedia](https://en.wikipedia.org/wiki/Load_balancing_%28computing%29),

> In computing, load balancing improves the distribution of workloads across multiple computing resources, such as computers, a computer cluster, network links, central processing units, or disk drives.

#### Create a Classic Load Balancer

Choose a Classic Load balancer which is used for HTTP, HTTPS and TCP traffic.

![](/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_12.png#layoutTextWidth)![](/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_13.png#layoutTextWidth)

Follow the wizard to complete the **Load Balancer** settings.

#### Define Load Balancer Properties

Define its name and protocol as depicted in the section below.

![](/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_14.png#layoutTextWidth)

#### Assign Security Groups

Assign security groups to the load balancer. Create a new security group for your load balancer that allows traffic from any IP address.

![](/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_15.png#layoutTextWidth)

Set Load balancer’s security group to the newly created one.

![](/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_16.png#layoutTextWidth)

#### Setup Health Checks

Setup health checks. Again you and go ahead with the default options as of now. We can revisit these settings later.

![](/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_17.png#layoutTextWidth)

#### Map with an EC2 instance

Choose the EC2 instance that you created earlier for the **load balancer**.

![](/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_18.png#layoutTextWidth)

#### Create tags

Tags are optional and help to identify your resources.

![](/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_19.png#layoutTextWidth)

All done! Your load balancer is now set up.

### Edit Inbound Rules for EC2 instance

Go back to your EC2 instance and add a new inbound rule for it. This will allow all traffic from the load balancer.

![](/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_20.png#layoutTextWidth)

Now your EC2 instance, hosting zone, and load balancer are all setup. We will go ahead and install a LAMP web server on our EC2 instance.

### Connect to your Instance

- Open an SSH client
- Locate your private key file and provide permissions

```
chmod 400 amazon_ec2_key.pem
```

- Connect to your EC2 instance

```
ssh -i "amazon_ec2_key.pem" ec2-user@ec2-13-127-42-0.ap-south-1.compute.amazonaws.com
```

### Install a LAMP webserver

- Check if software packages are up to date.

```
sudo yum update -y
```

- Install the packages for php and mysql.

```
sudo yum install -y httpd24 php70 mysql56-server php70-mysqlnd
```

- Start the http server

```
sudo service httpd start
```

- Create a `health.html` file in `cd/var/www/html` . Put any content in the file and save it.

```
nano health.html
```

- Edit health check settings for your load balancer.

![](/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_21.png#layoutTextWidth)

Now you should be able to see the test page when you hit the domain.

[http://aloudapp.in](http://aloudapp.in)

![](/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_22.png#layoutTextWidth)

#### Set File Permissions for your User

Run the following commands to set file permissions.

```
sudo usermod -a -G apache ec2-user
sudo chown -R ec2-user:apache /var/www
sudo chmod 2775 /var/www
find /var/www -type d -exec sudo chmod 2775 {} \;
find /var/www -type f -exec sudo chmod 0664 {} \;
```

![](/post/img/2017-12-03_web-hosting-using-php-and-mysql-on-aws_23.png#layoutTextWidth)

#### Test your LAMP server

Create a PHP file in Apache document root.

```
echo "<?php phpinfo(); ?>" > /var/www/html/phpinfo.php
```

Open [http://aloudapp.in/phpinfo.php](http://aloudapp.in/phpinfo.php) to test it.

#### Secure the MySQL server

- Start MySQL server.

```
sudo service mysqld start
```

- Run `mysql_secure_installation` and set root password, remove anonymous user accounts, disable remote root login, remove the test database, reload the privilege tables.

```
sudo mysql_secure_installation
```

### Install phpMyAdmin

Run the following commands to setup phpMyAdmin.

```
sudo yum install php70-mbstring.x86_64 php70-zip.x86_64 -y

sudo service httpd restart

cd /var/www/html

wget https://www.phpmyadmin.net/downloads/phpMyAdmin-latest-all-languages.tar.gz

tar -xvzf phpMyAdmin-latest-all-languages.tar.gz

mv phpMyAdmin-4.7.6-all-languages phpMyAdmin

sudo service mysqld start
```

Special thanks to [Chetan Gulati](https://www.facebook.com/chetangulati27) who helped me set it up.

**All done!**Now your web server is up and running. We will cover the configuration of YOURLS on this web server in the next story.

You can buy me a coffee if this post really helped you learn something or fix a nagging issue!

* * *
Written on December 3, 2017 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/web-hosting-using-php-and-mysql-on-aws-95bd5df0bd75)
