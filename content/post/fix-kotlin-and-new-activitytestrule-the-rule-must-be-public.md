---
title: "Fix Kotlin and new ActivityTestRule : The @Rule must be public"
author: "Vivek Maskara"
date: 2018-02-27T12:54:19.772Z
lastmod: 2021-10-29T21:23:23-07:00

description: ""

subtitle: ""

categories: [Android]

tags:
 - Android
 - Espresso
 - Testing
 - Kotlin
 - Android App Development


image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2018-02-27_fix-kotlin-and-new-activitytestrule-the-rule-must-be-public_0.png"
 - "/post/img/2018-02-27_fix-kotlin-and-new-activitytestrule-the-rule-must-be-public_1.png"


aliases:
- "/fix-kotlin-and-new-activitytestrule-the-rule-must-be-public-f0c5c583a865"

---

I wrote a simple activity in Koltin which takes a user name as input and simply displays it in the `TextView`. Here’s how the activity looks like.

![](/post/img/2018-02-27_fix-kotlin-and-new-activitytestrule-the-rule-must-be-public_0.png#layoutTextWidth)

And the code for the activity is,

```
class MainActivity : AppCompatActivity() {

    var button: Button? = null
    var userNameField: EditText? = null
    var displayUserName: TextView? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        initViews()
    }

    private fun initViews() {
        button = this.findViewById(R.id.set_user_name)
        userNameField = this.findViewById(R.id.name_field)
        displayUserName = this.findViewById(R.id.display_user_name)

        this.button!!.setOnClickListener({
            displayUserName!!.text ="Hello ${userNameField!!.text}!"
        })
    }
}
```

Now let's try to write a test for this activity which verifies if the `TextView` displays the name correctly or not. My test class looks have the following code:

```
@RunWith(AndroidJUnit4::class)
class MainActivityTest {

    @Rule var activityTestRule = ActivityTestRule(MainActivity::class.java)

    @Before
    fun setUp() {
    }

    @Test
    fun setUserName() {
        onView(withId(R.id.name_field)).perform(typeText("Vivek Maskara"))
        onView(withId(R.id.set_user_name)).perform(click())
        onView(withText("Hello Vivek Maskara!")).check(matches(isDisplayed()))
    }
}
```

When I run the test it fails with the following stack trace:

```
org.junit.internal.runners.rules.ValidationError: The @Rule 'activityTestRule' must be public.
at org.junit.internal.runners.rules.RuleMemberValidator$MemberMustBePublic.validate(RuleMemberValidator.java:222)
at org.junit.internal.runners.rules.RuleMemberValidator.validateMember(RuleMemberValidator.java:99)
at org.junit.internal.runners.rules.RuleMemberValidator.validate(RuleMemberValidator.java:93)
at org.junit.runners.BlockJUnit4ClassRunner.validateFields(BlockJUnit4ClassRunner.java:196)
at org.junit.runners.BlockJUnit4ClassRunner.collectInitializationErrors(BlockJUnit4ClassRunner.java:129)
.
.
.
```

The error is weird as the `@Rule` is already public.

JUnit allows to provide rules through a test class field or a getter method. What you annotated is in Kotlin a property though, which JUnit won’t recognise.

```
@Rule var activityTestRule = ActivityTestRule(MainActivity::class.java)
```

The annotation processor supports [annotation targets](https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.annotation/-target/index.html) and by default it uses the `property` target. In the above snippet it takes it as a `property` target unless specified otherwise.

#### First Approach

You can annotate the property getter however, which is also public and thus satisfies JUnit requirements for a rule getter:

```
@get:Rule var activityTestRule = ActivityTestRule(MainActivity::class.java)
```

#### Second Approach

Kotlin also allows compile properties to fields on the JVM, in which case the annotations and modifiers apply to the generated field. This is done using Kotlin’s [`@JvmField`](http://twitter.com/JvmField) property annotation. The fix is to add a `@JvmField` annotation to it. Read more about java annotations [here](https://kotlinlang.org/docs/reference/annotations.html#java-annotations).

This works perfectly.

```
@Rule @JvmField var activityTestRule = ActivityTestRule(MainActivity::class.java)
```

I prefer the second approach and finally my test class looks like:

```
@RunWith(AndroidJUnit4::class)
class MainActivityTest {

    @Rule @JvmField var activityActivityTestRule = ActivityTestRule(MainActivity::class.java)

    @Before
    fun setUp() {
    }

    @Test
    fun setUserName() {
        onView(withId(R.id.name_field)).perform(typeText("Vivek Maskara"))
        onView(withId(R.id.set_user_name)).perform(click())
        onView(withText("Hello Vivek Maskara!")).check(matches(isDisplayed()))
    }
}
```

You can buy me a coffee if this post really helped you learn something or fix a nagging issue!

* * *
Written on February 27, 2018 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/fix-kotlin-and-new-activitytestrule-the-rule-must-be-public-f0c5c583a865)
