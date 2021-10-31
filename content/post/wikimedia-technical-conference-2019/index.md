---
title: "Wikimedia Technical Conference 2019"
author: "Vivek Maskara"
date: 2019-11-22T03:03:28.188Z
lastmod: 2021-10-29T21:24:30-07:00

description: ""

subtitle: ""

categories: [Open Source]

tags:
 - Wikimedia
 - Conference

image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2019-11-22_wikimedia-technical-conference-2019_0.jpeg"
 - "/post/img/2019-11-22_wikimedia-technical-conference-2019_1.jpeg"
 - "/post/img/2019-11-22_wikimedia-technical-conference-2019_2.jpeg"
 - "/post/img/2019-11-22_wikimedia-technical-conference-2019_3.jpeg"
 - "/post/img/2019-11-22_wikimedia-technical-conference-2019_4.jpeg"


aliases:
- "/wikimedia-technical-conference-2019-1f4763c63336"

---

The [Wikimedia Technical Conference 2019](https://www.mediawiki.org/wiki/Wikimedia_Technical_Conference/2019), Atlanta concluded last week. I attended this conference for the first time, and it was an enjoyable experience for me. The central theme of the conference was around developer productivity, and several sessions were held covering different aspects of this theme.

### Personal Experience

This year’s conference was the first technical conference that I attended for Wikimedia. It was a very delightful experience for me.

![](/post/img/2019-11-22_wikimedia-technical-conference-2019_0.jpeg#layoutTextWidth)

I got to attend and participate in some very insightful sessions. I met with a lot of people from the foundation and volunteers, and it was great interacting with them.

### Commons Android app Team’s Participation

Our team took part in the session where **Best practices & useful methods for remote teams**was discussed.

![](/post/img/2019-11-22_wikimedia-technical-conference-2019_1.jpeg#layoutTextWidth)

Here is the presentation that we prepared to explain the process that our team follows.

### Takeaways

As most of our core team was also present at the conference, we decided to have a short discussion to incorporate our learnings from the various sessions we attended, into our team’s processes.

![](/post/img/2019-11-22_wikimedia-technical-conference-2019_2.jpeg#layoutTextWidth)![](/post/img/2019-11-22_wikimedia-technical-conference-2019_3.jpeg#layoutTextWidth)

Here’s a summary of the points that we discussed:

- Reevaluating the usage of Kotlin in the Commons app. As of now, the usage of Kotlin is limited to just unit test cases, and we are trying to evaluate if it would be good if it can be used in the `main` classes. Based on our discussion with other folks at the conference, we decided to restart the discussion and arrive at a consensus on it. You can participate in the discussion here:

[Evaluate usage of kotlin in the project · Issue #747 · commons-app/apps-android-commons](https://github.com/commons-app/apps-android-commons/issues/747 "https://github.com/commons-app/apps-android-commons/issues/747")

- We have decided to start monthly skill share sessions where we will pick a topic, and one of the team members will talk about it over a video call. Some of the issues that we have chosen for the first few sessions are MVP architecture, Kotlin, Unit testing, Dagger, and RxJava.
- We also agreed upon introducing some social connections amongst the team members and volunteers. We will have virtual coffee sessions every month, where we will talk about everything except work. :). Apart from these virtual coffee sessions, we will also create a new social channel on Zulip, where we will talk about non-work related stuff.
- We also discussed the need for focussing on integrated tests and UI tests for the app. So we have decided to make Espresso tests part of CI and also make it a priority to add more flows in the UI tests.
- In the next few months, we will also focus on integrating structured data into the commons app. We will build upon the work done by [Vanshika Arora ](https://medium.com/u/89df44efa52)and try to release structured data support as soon as possible for everyone.
- Lastly, we discussed validating if our app supports all languages that are supported by Android and MediaWiki.

### Conclusion

The event was quite delightful to me. Overall it was a very fruitful conference for our team, and we were able to deliberate on several issues, which are sometimes not possible to discuss remotely. Here’s the group picture of all the attendees of the conference.

![](/post/img/2019-11-22_wikimedia-technical-conference-2019_4.jpeg#layoutTextWidth)

Hopefully, the changes that we have discussed will help in improving our team’s productivity and make the app a lot more delightful for the users.

* * *
Written on November 22, 2019 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/wikimedia-technical-conference-2019-1f4763c63336)
