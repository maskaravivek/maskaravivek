---
title: "How To Improve Coverage for your Android App Using Mockito and Espresso — Part 1"
author: "Vivek Maskara"
date: 2018-05-16T13:16:13.601Z
lastmod: 2021-10-29T21:23:36-07:00

description: ""

subtitle: ""

categories: [Android, Testing]

tags:
 - Android
 - Kotlin
 - Unit Testing
 - Testing
 - Espresso
 - Mockito


image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2018-05-16_how-to-improve-coverage-for-your-android-app-using-mockito-and-espresso-part-1_0.jpeg"
 - "/post/img/2018-05-16_how-to-improve-coverage-for-your-android-app-using-mockito-and-espresso-part-1_1.png"


aliases:
- "/how-to-improve-coverage-for-your-android-app-using-mockito-and-espresso-185d3ab9bfc1"

---

![](/post/img/2018-05-16_how-to-improve-coverage-for-your-android-app-using-mockito-and-espresso-part-1_0.jpeg#layoutTextWidth)

In app development, a variety of use cases and interactions come up as one iterates the code. The app might need to fetch data from a server, interact with the device’s sensors, access local storage, or render complex user interfaces. The important thing to consider while writing tests is the units of responsibility that emerge as you design the new feature. The unit test should cover all possible interactions with the unit including standard interactions and exceptional scenarios. In this article, we will cover the fundamentals of testing and frameworks like Mockito and Espresso that developers can use to write unit tests. In the [second part](/how-to-improve-coverage-for-your-android-app-using-mockito-and-espresso-part-2-f4c2ac57bce8) of the article we will gets hand on and write a sample application with unit and instrumentation tests.

### Fundamentals Of Testing

A typical unit test contains 3 phases.

- First, the unit test initializes a small piece of an application it wants to test.
- Then it applies some stimulus to the system under test usually by calling a method on it
- Finally, it observes the resulting behavior.

If the observed behavior is consistent with the expectations, the unit test passes, otherwise, it fails, indicating that there is a problem somewhere in the system under test. These three unit test phases are also known as **Arrange**, **Act,** and **Assert**, or simply AAA. The app should ideally include three categories of tests: small medium and large.

- **Small tests** comprise of unit tests that mock every major component and run quickly in isolation
- **Medium tests** are integration tests that integrate several components and run on emulators or real devices
- **Large tests** are integration and UI tests that run by completing a UI workflow and ensure that the key end-user tasks work as expected.

Writing small tests allows you to address failures quickly but it’s difficult to gain confidence that a passing test allows your app to work. It’s important to have tests from all categories in the app, though the proportion of each category can vary from app to app. A good unit test should be **easy to write**, **readable**, **reliable** and **fast**.
 Here’s a brief introduction to Mockito and Espresso that make testing Android apps easier.

### Mockito

There are various mocking frameworks but the most popular of them all is [Mockito](http://site.mockito.org/)!

> Mockito is a mocking framework that tastes really good. It lets you write beautiful tests with a clean & simple API. Mockito doesn’t give you hangover because the tests are very readable and they produce clean verification errors.

Its fluent API separates pre-test preparation from post-test validation. Should the test fail, Mockito makes it clear to see where our expectations differ from reality! The library has everything you need to write complete tests.

### Espresso

Espresso helps you write concise, beautiful, and reliable Android UI tests. The code snippet below shows an example of an Espresso test. We will take up the same example again later in this tutorial when we talk in detail about instrumentation tests.

```
@Test
public void setUserName() {
    onView(withId(R.id.name_field)).perform(typeText("Vivek Maskara"));
    onView(withId(R.id.set_user_name)).perform(click());
    onView(withText("Hello Vivek Maskara!")).check(matches(isDisplayed()));
}
```

Espresso tests state expectations, interactions, and assertions clearly without the distraction of boilerplate content, custom infrastructure, or messy implementation details getting in the way. Whenever your test invokes `onView()`, Espresso waits to perform the corresponding UI action or assertion until the synchronization conditions are met ie.

- The message queue is empty
- No instances of `AsyncTask` is currently executing a task
- The idling resources are idle

These checks ensure that the test results are reliable and dependable.

### Writing Testable Code

Unit testing Android apps is difficult and sometimes impossible. Only and only a good design can help you making unit testing easier. Here are some of the concepts that are important for writing testable code.

### Avoid Mixing Object Graph Construction with Application Logic.

In a test, you want to instantiate the class under test and apply some stimulus to the class and assert that the expected behavior was observed. Make sure that the class under test doesn’t instantiate other objects and those objects do not instantiate more objects and so on. In order to have a testable code-base, your application should have two kinds of classes.

- The factories, these are full of the “new” operators and are responsible for building the object graph of your application.
- The application logic classes which are devoid of the “new” operator and are responsible for doing work.

### Constructors Should Not Do Any Work

The most common operation you will do in tests is the instantiation of object graphs, so make it easy on yourself and make the constructors do no work other than assigning all of the dependencies into the fields. Doing work in the constructor will not just affect the direct tests of the class, but will also affect related tests which try to instantiate your class indirectly.

### Avoid Static Methods Wherever Possible

The key to testing is the presence of places where you can divert the normal execution flow. Seams are needed so that you can isolate the unit of test. If you build an application with nothing but static methods you have procedural application. How much a static method will hurt from a testing point of view depends on where it is in your application call graph. A leaf method such as `Math.abs()` is not a problem since the execution call graph ends there. But if you pick a method in a core of your application logic than everything behind the method becomes hard to test, since there is no way to insert test doubles

### Avoid Mixing of Concerns

A class should be responsible for dealing with just one entity. Inside a class a method should be responsible for doing just one thing. For eg. `BusinessService` should be responsible just for talking to a `Business` and not `BusinessReceipts`. Moreover, a method in `BusinessService` could be `getBusinessProfile`, but a method such as `createAndGetBusinessProfile` would not be ideal for testing. [Solid design principles](https://en.wikipedia.org/wiki/SOLID_%28object-oriented_design%29?oldformat=true) must be followed for good design like:

- **S** — Single-responsibility principle
- **O** — The open-closed principle
- **L** — Liskov substitution principle
- **I** — Interface segregation principle
- **D** — Dependency Inversion Principle

We now have a fair understanding of why unit testing is important in an Android app and what are the best practices that could be followed to write testable code. In the [second part](/how-to-improve-coverage-for-your-android-app-using-mockito-and-espresso-part-2-f4c2ac57bce8) of this tutorial, we will write a sample application to put the above philosophies into action.

You can buy me a coffee if this post really helped you learn something or fix a nagging issue!

* * *
Written on May 16, 2018 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/how-to-improve-coverage-for-your-android-app-using-mockito-and-espresso-185d3ab9bfc1)
