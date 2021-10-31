---
title: "Android: Resolving Issue with Formatted Attribute while Translating String Resources"
author: "Vivek Maskara"
date: 2018-02-13T21:18:33.083Z
lastmod: 2021-10-29T21:23:22-07:00

description: ""

subtitle: ""

categories: [Android]

tags:
 - Android


aliases:
- "/android-resolving-issue-with-formatted-attribute-while-translating-string-resources-4357f2766928"

---

Take a look at the below string

```
<string name="notifications_talk_page_message" formatted="false">%s left a message on your %s page</string>
```

The string contains a format parameter, `%s` which can be replaced by a string value. This innocent looking string resource was a part of a [pull request](https://github.com/commons-app/apps-android-commons/pull/1089) that i submitted to [Wikimedia Commons Android app](https://github.com/commons-app/apps-android-commons). The changes were tested and finally merged to `master`. With new string resources in master, [translate wiki](https://translatewiki.net/) begin its magic of translating it to over 100 languages that the app currently supports.

The pipeline is setup to auto merge changes from translate wiki to master. The translation of the above string literal led to [compilation errors](https://github.com/commons-app/apps-android-commons/commit/72cddbf684ace2e5681a2404bbcfdba7eeaae95b). The problem is that apparently translatewiki.net cannot preserve the `formatted=”false”` attribute.

However, even if translatewiki.net supports it, they should be replaced with positional arguments anyway? I think sometimes translation may break the ordering of arguments. Checkout `argument_index` in [https://developer.android.com/reference/java/util/Formatter.html](https://developer.android.com/reference/java/util/Formatter.html) by ‘positional arguments’.

The [fix](https://github.com/commons-app/apps-android-commons/pull/1114/files) for this issue was to do away with the `formatted` parameter and use positional arguments.

```
<string name="notifications_talk_page_message">%1$s left a message on your %2$s page</string>
```

This change had to be made in all the string resource files to fix the issue.

* * *
Written on February 13, 2018 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/android-resolving-issue-with-formatted-attribute-while-translating-string-resources-4357f2766928)
