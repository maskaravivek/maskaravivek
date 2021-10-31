---
title: "How to Create a Subdomain in Amazon Route 53?"
author: "Vivek Maskara"
date: 2019-11-04T17:20:27.266Z
lastmod: 2021-10-29T21:24:29-07:00

description: ""

subtitle: ""

categories: [AWS]

tags:
 - AWS
 - Route53

image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2019-11-04_how-to-create-a-subdomain-in-amazon-route-53_0.png"
 - "/post/img/2019-11-04_how-to-create-a-subdomain-in-amazon-route-53_1.png"
 - "/post/img/2019-11-04_how-to-create-a-subdomain-in-amazon-route-53_2.jpeg"
 - "/post/img/2019-11-04_how-to-create-a-subdomain-in-amazon-route-53_3.jpeg"
 - "/post/img/2019-11-04_how-to-create-a-subdomain-in-amazon-route-53_4.png"
 - "/post/img/2019-11-04_how-to-create-a-subdomain-in-amazon-route-53_5.png"


aliases:
- "/how-to-create-a-subdomain-in-amazon-route-53-f14a995c4295"

---

![](/post/img/2019-11-04_how-to-create-a-subdomain-in-amazon-route-53_0.png#layoutTextWidth)

In this post, I will show you how to set up a new subdomain using Route 53 in 4 easy steps. The setup is straightforward and will take less than 2 minutes to have your subdomain up and running. :)

#### Step 1: Make a hosted zone for your original domain

Make sure that you have already created a hosted zone for your original domain. If you need guidance, you can refer to this blog post, where I set up a hosted zone for [aloudapp.in](http://aloudapp.in).

[Web Hosting using PHP and MySQL on AWS](/web-hosting-using-php-and-mysql-on-aws-95bd5df0bd75 "https://medium.com/@maskaravivek/web-hosting-using-php-and-mysql-on-aws-95bd5df0bd75")

**Step 2: Create a new hosted zone for your subdomain**

Click on **Create Hosted Zone**to create a new hosted zone. Enter the domain name as [api.aloudapp.in](http://api.aloudapp.in)and enter anything you want in the comment section. Set the type as **Public Hosted Zone**and click **Create.**

![](/post/img/2019-11-04_how-to-create-a-subdomain-in-amazon-route-53_1.png#layoutTextWidth)

**Step 3: Create a new NS record set for your original domain**

1. Click into your hosted zone, and select the recordset with type `NS`.

2. Copy the nameservers in the `value` box on the right.

![](/post/img/2019-11-04_how-to-create-a-subdomain-in-amazon-route-53_2.jpeg#layoutTextWidth)

3. Go back to your original hosted zone and click on **Create Record Set.**

4. Create a new NS record set with the name same as your subdomain and the value that you just copied. Click on **Create**to create it.

![](/post/img/2019-11-04_how-to-create-a-subdomain-in-amazon-route-53_3.jpeg#layoutTextWidth)

**Step 4: Create an A record set for your subdomain**

Return to your subdomain’s hosted zone and click on **Create Record Set**to create a new recordset. Leave the name empty and set the type as **A**. In the value field, you can enter any IP address that you want it to point to. Click on **Create**to create it.

![](/post/img/2019-11-04_how-to-create-a-subdomain-in-amazon-route-53_4.png#layoutTextWidth)

That’s it. Your subdomain should be up and running. :)

You can buy me a coffee if this post really helped you learn something or fix a nagging issue!

* * *
Written on November 4, 2019 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/how-to-create-a-subdomain-in-amazon-route-53-f14a995c4295)
