---
title: "How to Host Multiple WordPress sites on Cloud"
author: "Vivek Maskara"
date: 2020-03-24T02:26:05.000Z
lastmod: 2021-10-29T21:25:13-07:00

description: ""

subtitle: ""

categories: [AWS]

tags:
 - Wordpress
 - AWS
 - Hosting

image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2020-03-24_how-to-host-multiple-wordpress-sites-on-cloud_0.jpeg"
 - "/post/img/2020-03-24_how-to-host-multiple-wordpress-sites-on-cloud_1.jpg"
 - "/post/img/2020-03-24_how-to-host-multiple-wordpress-sites-on-cloud_2.jpg"


aliases:
- "/how-to-host-multiple-wordpress-sites-on-cloud-e77917aeb1b8"

---

![](/post/img/2020-03-24_how-to-host-multiple-wordpress-sites-on-cloud_0.jpeg#layoutTextWidth)

I have a few WordPress blogs, a couple of PHP based websites and a flask application. I recently decided to host all my websites in one place. Until recently I was using Amazon EC2 for some of these websites but it was proving to be quite costly so I decided to switch to Google Cloud.

### Setup Your Instance

I have used a LAMP Stack instance from Google’s VM marketplace but honestly, you can use any instance you like. Using the LAMP stack will simply give your Apache HTTP server, MySQL and PHP installed beforehand.

![](/post/img/2020-03-24_how-to-host-multiple-wordpress-sites-on-cloud_1.jpg#layoutTextWidth)

### Install packages

We will need to install a few packages to get things going. These packages will make PHP and MySQL available on our box.

```
sudo apt update sudo apt install php libapache2-mod-php mysql-server php-mysql
```

### Configure Apache Server

Next, we need to configure our Apache server. Browse to the following folder and create the config files there:

```
cd /etc/apache2/sites-available/
```

**Website 1**

This is the config file for my first website. I have named it `maskaravivek.com.conf`

```
<VirtualHost *:80>
     ServerAdmin admin@maskaravivek.com
     DocumentRoot /var/www/html/maskaravivek.com/
     ServerName maskaravivek.com
     ServerAlias www.maskaravivek.com

<Directory /var/www/html/maskaravivek.com/>
        Options Indexes FollowSymLinks MultiViews
        AllowOverride All
        Order allow,deny
        allow from all
     </Directory>

ErrorLog ${APACHE_LOG_DIR}/error.log
     CustomLog ${APACHE_LOG_DIR}/access.log combined
</VirtualHost>
```

**Note:**The config points to the `/var/www/html/maskaravivek.com/` directory. This is the place where the WordPress files for my first website reside.

**Website 2**

This is the config file for my first website. I have named it `windowsapptutorials.com.conf`

```
<VirtualHost *:80>
     ServerAdmin admin@windowsapptutorials.com
     DocumentRoot /var/www/html/windowsapptutorials.com/
     ServerName windowsapptutorials.com
     ServerAlias www.windowsapptutorials.com

<Directory /var/www/html/windowsapptutorials.com/>
        Options Indexes FollowSymLinks MultiViews
        AllowOverride All
        Order allow,deny
        allow from all
     </Directory>

ErrorLog ${APACHE_LOG_DIR}/error.log
     CustomLog ${APACHE_LOG_DIR}/access.log combined
</VirtualHost>
```

The config points to the `/var/www/html/windowsapptutorials.com/` directory.

**Enable Sites**

Next, let us enable both these websites using the following command.

```
sudo a2ensite maskaravivek.com.conf sudo a2ensite windowsapptutorials.com.conf
```

### Setup MySQL

Login into MySQL and run the following commands to create databases for both your websites.

**Website 1**

Create a database and user for Website 1. We create a user `maskaravivek_wordpress` and grant all database permissions to it.

```
CREATE DATABASE maskaravivek_wordpress;
GRANT SELECT,INSERT,UPDATE,DELETE,CREATE,DROP,ALTER ON maskaravivek_wordpress.* TO maskaravivek_wordpress@localhost IDENTIFIED BY '<password>';
```

**Website 2**

Create a database and user for Website 1. We create a user windows_app_tutorials_wordpress and grant all database permissions to it.

```
CREATE DATABASE windows_app_tutorials_wordpress;

GRANT SELECT,INSERT,UPDATE,DELETE,CREATE,DROP,ALTER ON windows_app_tutorials_wordpress.* TO windows_app_tutorials_wordpress@localhost IDENTIFIED BY '<password>'; 
```

Finally, run the following command so that the changes come into effect.

```
FLUSH PRIVILEGES; 
quit
```

### Install and Configure Wordpress

We need multiple copies of Wordpress on our box so we will download the latest `tar` of Wordpress and copy it to the respective directories specified by the Apache configs.

**Download**

Use the following commands to download Wordpress to the`tmp` folder and copy it to the respective locations.

```
cd /tmp/ && wget http://wordpress.org/latest.tar.gz 
tar -xzvf latest.tar.gz 
sudo cp -R wordpress/* /var/www/html/maskaravivek.com 
sudo cp -R wordpress/* /var/www/html/windowsapptutorials.com
```

**Configure Wordpress**

Next, copy the sample configs to the `wp-config.php` file for both the websites.

**Website 1**

```
sudo cp /var/www/html/maskaravivek.com/wp-config-sample.php /var/www/html/maskaravivek.com/wp-config.php 
```

**Website 2**

```
sudo cp /var/www/html/windowsapptutorials.com/wp-config-sample.php /var/www/html/windowsapptutorials.com/wp-config.php
```

Next, open the `wp-config.php` files one by one and update the database connection properties for both the websites.

```
sudo vim /var/www/html/maskaravivek.com/wp-config.php 
```

In the `wp-config.php` you just need to update the following 4 properties.

```
// ** MySQL settings - You can get this info from your web host ** // /** The name of the database for WordPress */ 
define('DB_NAME', 'examplecomdb'); 
/** MySQL database username */ 
define('DB_USER', 'examplecomuser'); 
/** MySQL database password */ 
define('DB_PASSWORD', 'type_password_here'); 
/** MySQL hostname */ define('DB_HOST', 'localhost');
```

Similarly, update the config for 2nd website.

```
sudo vim /var/www/html/windowsapptutorials.com/wp-config.php
```

### Final Touches

Next, update permissions for `/var/www/html` folder to make it executable.

```
sudo chown -R www-data:www-data /var/www/html 
sudo chmod -R 755 /var/www/html
```

Finally, run the following commands to update and restart the Apache Web service.

```
sudo apachectl configtest 
sudo a2enmod rewrite 
sudo systemctl restart apache2.service
```

That’s it. Both your websites are now setup. Browse to the following location to access your WordPress website.

```
http://<ip_address>/windowsapptutorials.com
```

This will redirect you to the Wordpress setup screen. Just follow the wizard to complete your setup.

![](/post/img/2020-03-24_how-to-host-multiple-wordpress-sites-on-cloud_2.jpg#layoutTextWidth)

* * *
Written on March 24, 2020 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/how-to-host-multiple-wordpress-sites-on-cloud-e77917aeb1b8)
