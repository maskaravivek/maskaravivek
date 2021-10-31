---
title: "Working with Recycler Views in Espresso Tests"
author: "Vivek Maskara"
date: 2020-03-10T19:09:27.984Z
lastmod: 2021-10-29T21:24:48-07:00

description: ""

subtitle: ""

categories: [Android]

tags:
 - Espresso
 - Android
 - Android Testing
 - Ui Testing
 - Recyclerview


image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2020-03-10_working-with-recycler-views-in-espresso-tests_0.jpeg"
 - "/post/img/2020-03-10_working-with-recycler-views-in-espresso-tests_1.png"
 - "/post/img/2020-03-10_working-with-recycler-views-in-espresso-tests_2.png"
 - "/post/img/2020-03-10_working-with-recycler-views-in-espresso-tests_3.png"


aliases:
- "/working-with-recycler-views-in-espresso-tests-6da21495182c"

---

![](/post/img/2020-03-10_working-with-recycler-views-in-espresso-tests_0.jpeg#layoutTextWidth)

In this short post, I will introduce you to a few utility methods that might be useful while working with `RecyclerView` in Espresso Tests.

### Introduction

In this post, I am assuming that you are already using a RecyclerView in your app. Take a look at this post if you need help in using a RecyclerView with an Android Paged list:

[Using the Paged List with Boundary Callback in Android](/using-the-paged-list-with-boundary-callback-in-android-b784a4f6f693 "https://medium.com/@maskaravivek/using-the-paged-list-with-boundary-callback-in-android-b784a4f6f693")

If you are just getting started with UI tests I would suggest that you take a look at my previous posts on using Espresso tests in Android.

[How To Improve Coverage for your Android App Using Mockito and Espresso — Part 2](https://proandroiddev.com/how-to-improve-coverage-for-your-android-app-using-mockito-and-espresso-part-2-f4c2ac57bce8 "https://proandroiddev.com/how-to-improve-coverage-for-your-android-app-using-mockito-and-espresso-part-2-f4c2ac57bce8")

### Custom Recycler View actions

If you have a `RecyclerView.ViewHolder` you can use it to access the child items and perform any action on them. The basic idea is to define a custom view action and then use it. The custom `ViewAction` takes the following form:

```
fun customAction(id: Int): ViewAction {
    return object : ViewAction {
        override fun getConstraints(): Matcher<View>? {
            return null
        }

        override fun getDescription(): String {
            return "Action Description"
        }

        override fun perform(uiController: UiController, view: View) {
            val v = view.findViewById<View>(id) as View 
            // your action
        }
    }
}
```

Using the above template lets take a few examples.

**Click a Recycler View Item**

This is one of the most common scenarios while working with a `RecyclerView` where you are required to click an item. For eg. in the screen shown below, I need to click on the nth item.

![](/post/img/2020-03-10_working-with-recycler-views-in-espresso-tests_1.png#layoutTextWidth)

Define a custom `ViewAction` like shown below:

```
fun clickItemWithId(id: Int): ViewAction {
    return object : ViewAction {
        override fun getConstraints(): Matcher<View>? {
            return null
        }

        override fun getDescription(): String {
            return "Click on a child view with specified id."
        }

        override fun perform(uiController: UiController, view: View) {
            val v = view.findViewById<View>(id) as View
            v.performClick()
        }
    }
}
```

And use it in your `RecyclerView` in the following way:

```
onView(withId(R.id.recycler_view))
  .perform(RecyclerViewActions
  .actionOnItemAtPosition<MyAdapter.ViewHolder>(1,clickItemWithId(R.id.button)))
```

The following two scenarios might not be a very common use case but I came across this scenario while working with the Wikimedia Commons Android app where we have a `RecyclerView` with a spinner and text view. The user can click on the (+) icon to add a new item to the list.

![](/post/img/2020-03-10_working-with-recycler-views-in-espresso-tests_2.png#layoutTextWidth)

**Type text in Recycler View’s child item**

Define a custom `ViewAction` like shown below:

```
fun typeTextInChildViewWithId(id: Int, textToBeTyped: String): ViewAction {
    return object : ViewAction {
        override fun getConstraints(): Matcher<View>? {
            return null
        }

        override fun getDescription(): String {
            return "Click on a child view with specified id."
        }

        override fun perform(uiController: UiController, view: View) {
            val v = view.findViewById<View>(id) as EditText
            v.setText(textToBeTyped)
        }
    }
}
```

And use it in your `RecyclerView` in the following way:

```
onView(withId(R.id.my_recycler_view))
  .perform(RecyclerViewActions
    .actionOnItemAtPosition<MyAdapter.ViewHolder>(1, 
      typeTextInChildViewWithId(R.id.my_edit_text, "your text")))
```

**Select a child Spinner Item from Recycler View**

Define a custom `ViewAction` like shown below:

```
fun selectSpinnerItemInChildViewWithId(id: Int, position: Int): ViewAction {
    return object : ViewAction {
        override fun getConstraints(): Matcher<View>? {
            return null
        }

        override fun getDescription(): String {
            return "Click on a child view with specified id."
        }

        override fun perform(uiController: UiController, view: View) {
            val v = view.findViewById<View>(id) as AppCompatSpinner
            v.setSelection(position)
        }
    }
}
```

And use it in your `RecyclerView` in the following way:

```
onView(withId(R.id.my_recycler_view))
  .perform(RecyclerViewActions
    .actionOnItemAtPosition<MyAdapter.ViewHolder>(1, 
      selectSpinnerItemInChildViewWithId(R.id.spinner_item, 2)))
```

#### Click Recycler View Item without Referencing the View Holder

If you have a simple Recycler view without a custom view holder you can use try matching the first or nth-child view without referencing the view holder.

You can use the following function to get the first matching view if you have a scenario where multiple views on the screen match your `Matcher` condition. This could happen when you are working with a list/recycler view.

```
fun <T> first(matcher: Matcher<T>): Matcher<T>? {
    return object : BaseMatcher<T>() {
        var isFirst = true
        override fun matches(item: Any): Boolean {
            if (isFirst && matcher.matches(item)) {
                isFirst = false
                return true
            }
            return false
        }

        override fun describeTo(description: Description) {
            description.appendText("should return first matching item")
        }
    }
}
```

For eg., if you want to click on the first item of your recycler view(ie. `R.id.rv_categories`) then you could use the above method in the following way:

```
onView(allOf(isDisplayed(), 
  first(withParent(withId(R.id.rv_categories)))))
   .perform(click())
```

### Conclusion

These were just a few examples of working with `RecyclerView` while writing Espresso tests. Feel free to browse the Commons Android app repository on Github to check out other Espresso tests.

[commons-app/apps-android-commons](https://github.com/commons-app/apps-android-commons/tree/master/app/src/androidTest/java/fr/free/nrw/commons "https://github.com/commons-app/apps-android-commons/tree/master/app/src/androidTest/java/fr/free/nrw/commons")

You can buy me a coffee if this post really helped you learn something or fix a nagging issue!

* * *
Written on March 10, 2020 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/working-with-recycler-views-in-espresso-tests-6da21495182c)
