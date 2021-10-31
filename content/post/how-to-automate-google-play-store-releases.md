---
title: "How to Automate Google Play Store Releases"
author: "Vivek Maskara"
date: 2019-02-18T15:00:20.436Z
lastmod: 2021-10-29T21:24:10-07:00

description: ""

subtitle: ""

categories: [Android]

tags:
 - Android
 - Google Play Store
 - Travis Ci
 - AndroidDev
 - Android App Development


image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2019-02-18_how-to-automate-google-play-store-releases_0.png"
 - "/post/img/2019-02-18_how-to-automate-google-play-store-releases_1.png"
 - "/post/img/2019-02-18_how-to-automate-google-play-store-releases_2.png"
 - "/post/img/2019-02-18_how-to-automate-google-play-store-releases_3.png"
 - "/post/img/2019-02-18_how-to-automate-google-play-store-releases_4.png"
 - "/post/img/2019-02-18_how-to-automate-google-play-store-releases_5.png"
 - "/post/img/2019-02-18_how-to-automate-google-play-store-releases_6.png"
 - "/post/img/2019-02-18_how-to-automate-google-play-store-releases_7.png"
 - "/post/img/2019-02-18_how-to-automate-google-play-store-releases_8.png"
 - "/post/img/2019-02-18_how-to-automate-google-play-store-releases_9.png"
 - "/post/img/2019-02-18_how-to-automate-google-play-store-releases_10.png"
 - "/post/img/2019-02-18_how-to-automate-google-play-store-releases_11.png"
 - "/post/img/2019-02-18_how-to-automate-google-play-store-releases_12.png"
 - "/post/img/2019-02-18_how-to-automate-google-play-store-releases_13.png"


aliases:
- "/how-to-automate-google-play-store-releases-c4c2a4b19435"

---

![](/post/img/2019-02-18_how-to-automate-google-play-store-releases_0.png#layoutTextWidth)

Manual publishing of an app is time-consuming and involves a lot of repetitive steps that could be easily automated. Manual publishing involves these steps:

- Create a signed APK for your app.
- Log in to the Google Play developer account, select the app and create a release.
- Upload the APK with a change-log.
- Submit the update.

In this tutorial, I will outline how you can automatically upload APKs to the Google Play Store using Google Play Developer APIs. We will also see how you can use Travis CI to automatically increment the version, build the app, ship to the Play Store, and more.

### Prerequisites

Before we get started with the tutorial, make sure that you have the following things already set up.

- Google Play Developer Account.
- An app listed in your developer account.
- A CI tool for triggering automatic builds.

**Note:** I will use Travis CI for this tutorial as it is free to use with open source projects.

### Signing your Android APK

To make a release to the Play Store, you need to [sign the APK](https://developer.android.com/studio/publish/app-signing) using a Keystore. Be sure to have the Keystore details specified in your app’s `build.gradle` file. You can use Android Studio’s inbuilt option (**Build** > **Generate Signed APK**…) to generate a Keystore or manually specify the Keystore details in the Gradle file.

![](/post/img/2019-02-18_how-to-automate-google-play-store-releases_1.png#layoutTextWidth)

Once you have created a Keystore, specify its details in the app’s `build.gradle` file. Add this snippet to your `build.grade`’s `android` section:

```
android {
    ...

signingConfigs {        release {            storeFile file("someDirectory/my_keystore.jks")            storePassword "my_store_pass_here"            keyAlias "my_key_alias_here"            keyPassword "my_key_pass_here"        }    }

buildTypes {
        release {
            signingConfig signingConfigs.release
            minifyEnabled false
            proguardFiles getDefaultProguardFile('proguard-android.txt'), 'proguard-rules.pro'
        }
    }
}
```

After setting up the signing keys in the Gradle file, you can execute the following command to generate a signed version of the APK.

```
./gradlew :app:assembleRelease
```

For reference, you can check out the [commit](https://github.com/maskaravivek/Example-Release-Automation/commit/8aeffc9ea98203850ee228c0c6024488bdc098a6) for configuring Signing keys in the sample app. These configs would later be used to automate the signing process of the app. Also for security purposes, we will encrypt these configs to prevent its malicious use.

### Setup API Access for Automatic Publishing

The Google Play Developer Console provides API support for you to be able to push updates automatically. This ability allows you to trigger builds on your continuous integration server and have them uploaded the Play store for alpha or beta testing, as well as pushing to production directly.

**Link Google Play Android Developer project** Login into your Google Play Developer account and go into **Settings** > **Developer account** > **API access.** Here, link **Google Play Android Developer** under **Linked Project** section.

![](/post/img/2019-02-18_how-to-automate-google-play-store-releases_2.png#layoutTextWidth)

**Create an OAuth Client and Service Account** Next, you need to create an **OAuth Client** and a **Service Account**. Simply click on **CREATE OAUTH CLIENT** button, and it will create an **OAuth Client**. Clicking on **CREATE SERVICE ACCOUNT** will show a dialog indicating that you need to visit **Google API** Console and manually create a **Service Account**.

![](/post/img/2019-02-18_how-to-automate-google-play-store-releases_3.png#layoutTextWidth)

Click on the **Create Service Account** button on the **Google API Console** page and set the service account name.

![](/post/img/2019-02-18_how-to-automate-google-play-store-releases_4.png#layoutTextWidth)

Click on **Create Key** button to create a new private key and select P12 as key type. Download the P12 file while contains the service account details.

![](/post/img/2019-02-18_how-to-automate-google-play-store-releases_5.png#layoutTextWidth)

This `.p12` file will be required for authenticating with Google Play Access API later on. Recheck the **API Access** page to make sure that the created service account appears on that page.

![](/post/img/2019-02-18_how-to-automate-google-play-store-releases_6.png#layoutTextWidth)

Click on Grant Access and make sure the checkboxes for Edit store listing, pricing & distribution, Manage Production APKs and Manage Alpha & Beta APKs are checked.

![](/post/img/2019-02-18_how-to-automate-google-play-store-releases_7.png#layoutTextWidth)

### Include the Gradle Plugin in the Project

Now that API access has been enabled for the app, we can use it to publish our app using these APIs. Fortunately, there is already an open source [Gradle Play Publisher](https://github.com/Triple-T/gradle-play-publisher) plugin, that can be used to upload your App Bundle or APK and other app details to the Google Play Store.

Add the following in the top level Gradle file `build.gradle` file:

```
buildscript {
    repositories {
        jcenter()
    }
    
    dependencies {
      // ...
      classpath 'com.github.triplet.gradle:play-publisher:2.0.0-rc1'
    }
}
```

Add the following to the top of your `app/build.gradle` file:

```
apply plugin: 'com.android.application'
apply plugin: 'com.github.triplet.play'

android {
    signingConfigs {
        release {            storeFile file("someDirectory/my_keystore.jks")            storePassword "my_store_pass_here"            keyAlias "my_key_alias_here"            keyPassword "my_key_pass_here"        }
    }

buildTypes {
        release {
            signingConfig signingConfigs.release
            minifyEnabled false
            proguardFiles getDefaultProguardFile('proguard-android.txt'), 'proguard-rules.pro'
        }
    }
}

play {    track = "alpha"    userFraction = 1    serviceAccountEmail = "SERVICE_ACCOUNT_NAME"    serviceAccountCredentials = file("../play.p12")

resolutionStrategy = "auto"    outputProcessor { // this: ApkVariantOutput        versionNameOverride = "$versionNameOverride.$versionCode"    }}
```

Now, that you have setup Gradle Play Publisher plugin to your project, you can check what new Gradle tasks have been added. Execute `./gradlew tasks` to review the tasks added by the plugin.

![](/post/img/2019-02-18_how-to-automate-google-play-store-releases_8.png#layoutTextWidth)

We can finally publish the app using this plugin by simply executing this task:

```
./gradlew publishReleaseApk
```

Check out the [commit](https://github.com/maskaravivek/Example-Release-Automation/commit/9ea8383339af648f73976e1fea3c51b4c6a17fc0) in the sample app to see changes done for Gradle Publisher Plugin integration. Note that if you try executing this task for an unreleased app, it will throw an error as shown in the screenshot.

![](/post/img/2019-02-18_how-to-automate-google-play-store-releases_9.png#layoutTextWidth)

This concludes the steps required for eliminating the manual release process. You still need to run this task for the actual release to happen. In the next, section we will go through the steps required for CI integration. The steps mentioned below are specific to Travis CI, but the process would be similar for other popular CIs like Jenkins or CircleCI.

### CI Integration

To have Travis build a signed, release version of the APK, it needs the Keystore details. The challenge is to be able to do this securely- without adding the Keystore file or any other information in the source control or having them visible to anyone publicly.

Achieving automated releases requires the following:

- Add an encrypted version of the Keystore file to the repository
- Store Keystore’s credentials securely
- Build and deploy a signed, release version of the APK when new code is pushed to the repository

### Prerequisites

**Setup the initial Travis Config** Before, we dive into configuring Travis for automated releases, make sure you have `.travis.yml` configured for your project. My config looks like the snippet shown below:

```
language: android
addons:
    apt:
    packages:
    - w3m
env:
    global:
    - ANDROID_TARGET=android-22
    - ANDROID_ABI=armeabi-v7a
    - ADB_INSTALL_TIMEOUT=12
jdk:
- oraclejdk8
android:
    components:
    - tools
    - platform-tools
    - build-tools-28.0.3
    - extra-google-m2repository
    - extra-android-m2repository
    - android-22
    - android-28
    licenses:
    - android-sdk-license-.+
script:
- "./gradlew clean build"
```

Check out the [initial Travis config](https://github.com/maskaravivek/Example-Release-Automation/commit/9c720a385fc33c85e30fd210ad277ea70fe30204) in the sample app.

**Install and Log** ******in to Travis CI Command Line Tools** Firstly, install Travis Command line tools by executing:

```
gem install travis
```

Next, login to Travis CI using:

```
travis login --com
```

If you are still using [travis-ci.org](http://travis-ci.org/), you need to use `--org`.

### Store Keystore and Service Account Key Files Securely with Travis

We already have our Keystore and Service Account key setup, and now we need to add them to Travis. We will be using its [File Encryption](https://docs.travis-ci.com/user/encrypting-files/#encrypting-multiple-files) functionality for it. Initially, we put the `keystore.jks` and `service-account-key.p12` files in the root folder of the app. Encrypting a single file is quite simple and can be achieved by executing a single command.

```
travis encrypt-file keystore.jks --add
```

In case of multiple files, Travis CLI has an issue which overrides the existing secure environment variables when the command is invoked for a different file. So, as a workaround, we will first create an archive of the sensitive files and then store it on Travis. Run the following commands for it:

```
tar cvf secrets.tar keystore.jks service-account-key.p12
```

You’ll see something like the following output.

![](/post/img/2019-02-18_how-to-automate-google-play-store-releases_10.png#layoutTextWidth)

Before proceeding to the next step, delete the original Keystore, Service Account Key and `secrets.tar` files from the repository folder. Now you're left with `secrets.tar.enc` - which is the encrypted version of the `secrets.tar` file. It can be safely committed to your repository. You will notice a `before_install` step added to your `.travis.yml` file.

```
before_install:
- openssl aes-256-cbc -K $encrypted_054692ad430f_key -iv $encrypted_054692ad430f_iv -in secrets.tar.enc -out secrets.tar -d
```

Also, your Travis CI Settings would start showing two encrypted environment variables corresponding to the Keystore file.

![](/post/img/2019-02-18_how-to-automate-google-play-store-releases_11.png#layoutTextWidth)

Notice that even we can’t see those values. They’re secret and only available to the Travis Virtual Machine when it works on your app. The command in the `before_install` step runs and decrypts the `secrets.tar.enc` file to generate the `secrets.tar`. Add another step to unarchive the `tar` file.

```
- tar xvf secrets.tar
```

### Store the Credentials Securely with Travis

Travis also requires the credentials to create a signed APK. Go to the Travis web console settings page and add your parameters. Ensure that the Display Value in build log option is toggled off else they would be printed to build logs.

![](/post/img/2019-02-18_how-to-automate-google-play-store-releases_12.png#layoutTextWidth)

When you build the app in the Travis environment, `key_alias`, `key_password`, and `keystore_password` would be available as system variables.

### Update Gradle File to Use Environment Variables

Next, you need to update the app’s `build.gradle` file to use these environment variables while building on CI. Note, that the secure environment variables(decrypted Keystore and `p12` files) are not available for Pull Requests. We will add a check in our Gradle file to determine if the environment variables can be used or not.

```
apply plugin: 'com.android.application'

def isRunningOnTravisAndIsNotPRBuild = System.getenv("CI") == "true" && file('../play.p12').exists()

if(isRunningOnTravisAndIsNotPRBuild) {
    apply plugin: 'com.github.triplet.play'
}

android {
    .
    .

signingConfigs {
        release
    }

.
    .

if (isRunningOnTravisAndIsNotPRBuild) {
        signingConfigs.release.storeFile = file("../keystore.jks")        signingConfigs.release.storePassword = System.getenv("keystore_password")        signingConfigs.release.keyAlias = System.getenv("key_alias")        signingConfigs.release.keyPassword = System.getenv("key_password")    }}

.
.

if (isRunningOnTravisAndIsNotPRBuild) {
    play {
        track = "alpha"
        userFraction = 1
        serviceAccountEmail = "test@api-522342-837358.iam.gserviceaccount.com"
        serviceAccountCredentials = file("../service-account-key.p12")

resolutionStrategy = "auto"
        outputProcessor { // this: ApkVariantOutput
            versionNameOverride = "$versionNameOverride.$versionCode"
        }
    }
}
```

`System.getenv` is one of the default variables Travis sets for us, and it lets us determine if the build is running on Travis CI or locally. If the build is running on Travis,

- Apply the Gradle Play Publisher Plugin
- Populate our signing config with the store file, password, key alias, and alias password.
- Add the `play` config for Play Store releases

### Update Travis Config to Release on Each Commit

Finally, update the Travis `.yml` file to run the `publishReleaseApk` task whenever a new commit is pushed to master. Add the following step in the `script` section of the `.travis.yml` file.

```
script:
- if [ "$TRAVIS_PULL_REQUEST" == "false" ] && [ "$TRAVIS_BRANCH" == "master" ]; then
  ./gradlew publishReleaseApk;
  fi
```

- `TRAVIS_PULL_REQUEST` determines if a Pull Request triggers the build. Remember, secure variables are not available for Pull Requests.
- `TRAVIS_BRANCH` determines whether the branch is `master`. We do not want to make a release whenever a commit gets pushed to another branch.

Check out the [commit](https://github.com/maskaravivek/Example-Release-Automation/commit/3cc6dd7e64de30c56a3956cc63116b6af00bb837) in the sample app to see changes done for encrypting and storing the keys on Travis. You are all set now! Commit your changes and push to master so that a build gets triggered on Travis. Once the build completes successfully, an APK gets uploaded to alpha track on Google Play Store.

### Conclusion

We have successfully eliminated all the manual steps required to make a Play Store release. Now, Travis can automatically publish a signed APK on every commit to master. We were able to quickly consume the Google Play Developer APIs owing to the open-source Gradle Play Publisher plugin. Also, we went through the steps required to store Keystore and Service Account keys with Travis securely.

We recently automated our `alpha` releases for [Wikimedia Commons Android app](https://github.com/commons-app/apps-android-commons) and it has immensely helped us in making around automatic 100 releases in December 2018 and collecting feedback from our Alpha users. It would have been quite difficult to make 100 manual releases and would have hampered the speed of closed group testing. For any fast-paced development environment, it is quite essential to eliminate the manual steps to boost developer productivity.

You can find all the code used in this article in this [example project on Github](https://github.com/maskaravivek/Example-Release-Automation).

You can buy me a coffee if this post really helped you learn something or fix a nagging issue!

* * *
Written on February 18, 2019 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/how-to-automate-google-play-store-releases-c4c2a4b19435)
