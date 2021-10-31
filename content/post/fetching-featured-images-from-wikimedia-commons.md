---
title: "Fetching Featured Images from Wikimedia Commons"
author: "Vivek Maskara"
date: 2018-05-12T14:09:54.062Z
lastmod: 2021-10-29T21:23:35-07:00

description: ""

subtitle: ""

tags: [Open Source, Wikimedia Commons]
categories: [Open Source]

image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2018-05-12_fetching-featured-images-from-wikimedia-commons_0.jpeg"
 - "/post/img/2018-05-12_fetching-featured-images-from-wikimedia-commons_1.jpeg"


aliases:
- "/fetching-featured-images-from-wikimedia-commons-f565c81e908"

---

![](/post/img/2018-05-12_fetching-featured-images-from-wikimedia-commons_0.jpeg#layoutTextWidth)

Wikimedia Commons features many images regularly on its main page. As described by Commons,

> **Featured pictures** are images from highly skilled [photographers](https://commons.wikimedia.org/wiki/Commons:Meet_our_photographers "Commons:Meet our photographers") and [illustrators](https://commons.wikimedia.org/wiki/Commons:Meet_our_illustrators "Commons:Meet our illustrators") that the Wikimedia Commons community has chosen as some of the highest quality on the site.

Getting an image featured is quite difficult as the statistics suggest.

> There are currently 11,192 of such images in the Commons repository which is roughly 0.024% of the available images (46,591,843).

In the Wikimedia Commons Android app, we thought [showing these featured images](https://github.com/commons-app/apps-android-commons/issues/324) in the app would be a good addition as it would:

- Show users what a great Commons picture is
- Inspire/motivate users to take part in the same project
- Let users hope that one day their picture will be shown at that place (in fact, show them that taking great pictures is worth the effort)

I started exploring the API that would let us fetch featured images and then we could consume it in the app. I came across the [MediaWiki Generator API](https://commons.wikimedia.org/w/api.php?action=help&modules=query).

Check out the following API call that returns the first 10 features images from the API. [https://bit.ly/2KRXg4l](https://bit.ly/2KRXg4l)

Here’s a screenshot from the app showing the featured images from Commons:

![](/post/img/2018-05-12_fetching-featured-images-from-wikimedia-commons_1.jpeg#layoutTextWidth)

Let us discuss the parameters that the API takes:

- **action**: query which is common param for all MediaWiki API calls
- **generator**: we are going to generate a list of `categorymembers`
- **gcmtype**: this param’s value is `file` as we want to extract information of the Commons image files
- **gcmtitle**: this is name of the category under which all featured images on Commons are stored. `Category:Featured_pictures_on_Wikimedia_Commons`
- **prop:**Setting the value of `prop` to `imageinfo` adds the `imageinfo` block in the API response. It contains the `url` and `extmetadata` fields that are required for getting some of the properties.

We wanted to display all the images in a category instead of showing just the first 10. So we used the [continue query](https://www.mediawiki.org/wiki/API:Raw_query_continue) of MediaWiki API. To get more data, you use the `query-continue` value in the response.

> The `query-continue` node will contain a subnode for each module used in the query that needs continuation, and these subnodes will contain properties to be used when making the followup "continuation" query. Note that clients should not be depending on the particular property names given for continuation of any module or the format of the values returned for continuation, as these may change.

Continue query: [https://bit.ly/2Idkhgt](https://bit.ly/2Idkhgt)

Moreover, the implementation in generic enough to load images from any category of Commons.

```
CategoryImagesActivity.startYourself(this, "Page title", "Category name");
```

Check out the full implementation for this feature here:

[Integrate API for displaying featured images by maskaravivek · Pull Request #1456 ·…](https://github.com/commons-app/apps-android-commons/pull/1456 "https://github.com/commons-app/apps-android-commons/pull/1456")

The changes are now merged to master and you could take a pull of the code base and run it for yourself. :)

[commons-app/apps-android-commons](https://github.com/commons-app/apps-android-commons/ "https://github.com/commons-app/apps-android-commons/")

Make sure you give this post **50 claps**and **follow**meif you enjoyed this post and want to see more!

* * *
Written on May 12, 2018 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/fetching-featured-images-from-wikimedia-commons-f565c81e908)
