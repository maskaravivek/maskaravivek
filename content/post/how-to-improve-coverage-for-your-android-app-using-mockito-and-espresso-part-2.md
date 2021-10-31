---
title: "How To Improve Coverage for your Android App Using Mockito and Espresso — Part 2"
author: "Vivek Maskara"
date: 2018-05-20T16:01:50.392Z
lastmod: 2021-10-29T21:23:41-07:00

description: ""

subtitle: ""

categories: [Android]

tags:
 - Android
 - Mockito
 - Espresso
 - Unit Testing
 - Kotlin


image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2018-05-20_how-to-improve-coverage-for-your-android-app-using-mockito-and-espresso-part-2_0.jpeg"
 - "/post/img/2018-05-20_how-to-improve-coverage-for-your-android-app-using-mockito-and-espresso-part-2_1.png"
 - "/post/img/2018-05-20_how-to-improve-coverage-for-your-android-app-using-mockito-and-espresso-part-2_2.png"
 - "/post/img/2018-05-20_how-to-improve-coverage-for-your-android-app-using-mockito-and-espresso-part-2_3.png"
 - "/post/img/2018-05-20_how-to-improve-coverage-for-your-android-app-using-mockito-and-espresso-part-2_4.png"
 - "/post/img/2018-05-20_how-to-improve-coverage-for-your-android-app-using-mockito-and-espresso-part-2_5.gif"
 - "/post/img/2018-05-20_how-to-improve-coverage-for-your-android-app-using-mockito-and-espresso-part-2_6.png"
 - "/post/img/2018-05-20_how-to-improve-coverage-for-your-android-app-using-mockito-and-espresso-part-2_7.png"


aliases:
- "/how-to-improve-coverage-for-your-android-app-using-mockito-and-espresso-part-2-f4c2ac57bce8"

---

![](/post/img/2018-05-20_how-to-improve-coverage-for-your-android-app-using-mockito-and-espresso-part-2_0.jpeg#layoutTextWidth)

In the [first part](/how-to-improve-coverage-for-your-android-app-using-mockito-and-espresso-185d3ab9bfc1) of this article we got an introduction about various frameworks available to us for writing tests for an Android app. We also saw some best practices that could be followed to write more testable code. Now lets write some actual code and then add unit and instrumentation tests for the same.

In the next few, sections we would be using examples from a really simple application that I built for the purpose of this tutorial. The app has an `EditText` that takes a username as input and displays the name in a `TextView` on click of a button. Feel free to take the complete [source code](https://github.com/maskaravivek/AndroidTestingExamples) for the project from Github. Here’s a screenshot of the app.

![](/post/img/2018-05-20_how-to-improve-coverage-for-your-android-app-using-mockito-and-espresso-part-2_1.png#layoutTextWidth)

### Writing Local Unit Tests

Unit tests can be run locally on the development machine without a device or an emulator. This testing approach is efficient because it helps you avoid the overhead of loading the target app and unit test code onto a physical device or emulator every time your test is run. In addition to Mockito, you will also need to configure the testing dependencies for your project to use the standard APIs provided by the JUnit 4 framework.

### Setting up the Development Environment

Start by adding a dependency on JUnit4 in your project. The dependency is of type `testImplementation` which means that the dependencies are only required to compile the test source of the project.

```
testImplementation 'junit:junit:4.12'
```

We will also need Mockito library to make interacting with Android dependencies easier.

```
testImplementation "org.mockito:mockito-core:$MOCKITO_VERSION"
```

Make sure to sync the project after adding the dependency. Android studio should have created the folders structure for unit tests by default, if not make sure the following directory structure exists.

```
<Project Dir>/app/src/test/java/com/maskaravivek/testingExamples
```

### Creating your First Unit Test

Suppose you want to test the `displayUserName` function in the `UserService`. For the sake of simplicity, the function simply formats the input and returns it back. In a real-world application, it could make a network call to fetch the user profile and return the user’s name.

```
@Singleton
class UserService @Inject
constructor(private var context: Context) {
fun displayUserName(name: String): String {
        val userNameFormat = context.getString(R.string.display_user_name)
        return String.format(Locale.ENGLISH, userNameFormat, name)
    }
}
```

We will start by creating a `UserServiceTest` class in our test directory. The `UserService` class uses `Context` which needs to be mocked for the purpose of testing. Mockito provides a `@Mock` notation for mocking objects which can be used as follows,

```
@Mock internal var context: Context? = null
```

Similarly, you need to mock all dependencies required to construct the instance of the `UserService` class. Before your test, you need to initialize these mocks and inject it into the `UserService` class.

- `@InjectMock` creates an instance of the class and injects the mocks that are marked with the annotations `@Mock` into it.
- `MockitoAnnotations.initMocks(this);` initializes fields annotated with Mockito annotations.

Here’s how it can be done.

```
class UserServiceTest {
@Mock internal var context: Context? = null
    @InjectMocks internal var userService: UserService? = null
@Before
    fun setup() {
        MockitoAnnotations.initMocks(this)
    }
}
```

Now you are done setting up your test class. Let’s add a test to this class that verifies the functionality of `displayUserName` function. Here's how the test looks,

```
@Test
fun displayUserName() {
    doReturn("Hello %s!").`when`(context)!!.getString(any(Int::class.java))
    val displayUserName = userService!!.displayUserName("Test")
    assertEquals(displayUserName, "Hello Test!")
}
```

The test uses a `doReturn().when()` statement to provide a response when a `context.getString()` is invoked. For any input integer, it will return the same result `"Hello %s!"`. We could have been more specific by making it return this response only for a particular string resource ID but for the sake of simplicity we are returning the same response to any input. Finally, here's how the test class looks,

```
class UserServiceTest {
        @Mock internal var context: Context? = null
        @InjectMocks internal var userService: UserService? = null
        @Before
        fun setup() {
            MockitoAnnotations.initMocks(this)
        }
     
        @Test
        fun displayUserName() {
            doReturn("Hello %s!").`when`(context)!!.getString(any(Int::class.java))
            val displayUserName = userService!!.displayUserName("Test")
            assertEquals(displayUserName, "Hello Test!")
        }
    }
```

### Running your Unit Tests

In order to run the unit tests, you need to make sure that Gradle is synchronized. In order to run a test click on the green play icon in the IDE.

![](/post/img/2018-05-20_how-to-improve-coverage-for-your-android-app-using-mockito-and-espresso-part-2_2.png#layoutTextWidth)

When the unit tests are run, successfully or otherwise, you should be able to see this in the run menu at the bottom of the screen.

![](/post/img/2018-05-20_how-to-improve-coverage-for-your-android-app-using-mockito-and-espresso-part-2_3.png#layoutTextWidth)

You are done with your first unit test!

### Writing Instrumentation Tests

Instrumentation tests are most suited for checking values of UI components when an activity is run, for instance in the above example we want to make sure that the `TextView` shows the correct username after the `Button` is clicked. They run on physical devices and emulators and can take advantage of the Android framework APIs and supporting APIs, such as the Android Testing Support Library. We'll use espresso to take actions on the main thread such as button clicks, text change etc.

### Setting up the Development Environment

Add a dependency on espresso:

```
androidTestImplementation 'com.android.support.test.espresso:espresso-core:3.0.1'
```

Instrumentation tests are created in an androidTest folder.

```
<Project Dir>/app/src/androidTest/java/com/maskaravivek/testingExamples
```

If you want to test a simple activity, you should create your test class in the same package as your Activity.

### Creating your First Instrumentation Test

Let’s start by creating a simple activity that takes a name as input and on a click of a button it sets displays the username. The activity looks like: The code for the above activity is quite simple.

```
class MainActivity : AppCompatActivity() {
var button: Button? = null
    var userNameField: EditText? = null
    var displayUserName: TextView? = null
override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        AndroidInjection.inject(this)
        setContentView(R.layout.activity_main)
        initViews()
    }
private fun initViews() {
        button = this.findViewById(R.id.set_user_name)
        userNameField = this.findViewById(R.id.name_field)
        displayUserName = this.findViewById(R.id.display_user_name)
this.button!!.setOnClickListener({
            displayUserName!!.text = "Hello ${userNameField!!.text}!"
        })
    }
}
```

To create a test for the `MainActivity` we will start by creating a `MainActivityTest` class under the androidTest directory. Add the `AndroidJUnit4` annotation to the class to indicate that the tests in this class will use the default Android test runner class.

```
@RunWith(AndroidJUnit4::class) class MainActivityTest {}
```

Next, add an `ActivityTestRule` to the class. This rule provides functional testing of a single Activity. During the duration of the test, you will be able to manipulate your Activity directly using the reference obtained from `getActivity()`.

```
@Rule @JvmField var activityActivityTestRule = ActivityTestRule(MainActivity::class.java)
```

Now that you are done setting up the test class, let’s add a test that verifies that the username is displayed by clicking the **Set User Name** button.

```
@Test
fun setUserName() {
    onView(withId(R.id.name_field)).perform(typeText("Vivek Maskara"))
    onView(withId(R.id.set_user_name)).perform(click())
    onView(withText("Hello Vivek Maskara!")).check(matches(isDisplayed()))
}
```

The above test is quite simple to follow. It first simulates typing some text in the `EditText`, performs the click action on the button and then checks whether the correct text is displayed in the `TextView`.

The final test class looks like,

```
@RunWith(AndroidJUnit4::class)
class MainActivityTest {
@Rule @JvmField var activityActivityTestRule = ActivityTestRule(MainActivity::class.java)
@Test
    fun setUserName() {
        onView(withId(R.id.name_field)).perform(typeText("Vivek Maskara"))
        onView(withId(R.id.set_user_name)).perform(click())
        onView(withText("Hello Vivek Maskara!")).check(matches(isDisplayed()))
    }
}
```

### Running your Instrumentation Tests

Just like for unit tests, click on the green play button in the IDE to run the test.

![](/post/img/2018-05-20_how-to-improve-coverage-for-your-android-app-using-mockito-and-espresso-part-2_4.png#layoutTextWidth)

On clicking the play button, the test version of the app is installed on the emulator or device and the test runs automatically on it.

![](/post/img/2018-05-20_how-to-improve-coverage-for-your-android-app-using-mockito-and-espresso-part-2_5.gif#layoutTextWidth)

### Intrumentation Testing Using Dagger, Mockito and Espresso

Espresso is one of the most popular UI testing frameworks with good documentation and community support. Mockito ensures that objects perform the actions that are expected from them. Mockito also works well with dependency injection libraries like Dagger. Mocking the dependencies will allow us to test a scenario in isolation. Until now our `MainActivity` isn't using any dependency injection and as a result of it, we were able to write our UI test very easily. To make things a bit interesting let’s inject `UserService` in the `MainActivity` and use it to get the text to be displayed.

```
class MainActivity : AppCompatActivity() {
var button: Button? = null
    var userNameField: EditText? = null
    var displayUserName: TextView? = null
@Inject lateinit var userService: UserService
override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        AndroidInjection.inject(this)
        setContentView(R.layout.activity_main)
        initViews()
    }
private fun initViews() {
        button = this.findViewById(R.id.set_user_name)
        userNameField = this.findViewById(R.id.name_field)
        displayUserName = this.findViewById(R.id.display_user_name)
this.button!!.setOnClickListener({
            displayUserName!!.text = userService.displayUserName(userNameField!!.text.toString())
        })
    }
}
```

With Dagger in the picture, we will have to set up a few things before we write instrumentation tests. Imagine the `displayUserName` function internally uses some API to fetch the details of the user. There should not be a situation in which tests do not pass due to server fault. To avoid such situation we can use dependency injection framework Dagger and for networking Retrofit.

### Setting up Dagger in the Application

We will quickly set up the basic modules and components required for Dagger. If you are not familiar with Dagger, check out [Google’s documentation](https://google.github.io/dagger/) on it. We will start adding dependencies for using Dagger in the `build.gradle` file.

```
implementation "com.google.dagger:dagger-android:$DAGGER_VERSION" 
implementation "com.google.dagger:dagger-android-support:$DAGGER_VERSION" 
implementation "com.google.dagger:dagger:$DAGGER_VERSION" 
kapt "com.google.dagger:dagger-compiler:$DAGGER_VERSION" 
kapt "com.google.dagger:dagger-android-processor:$DAGGER_VERSION"
```

Create a component in the Application class, add necessary modules which will be used in our project. We need to inject dependencies in the `MainActivity` of our app. We will add a `@Module` for injecting in the activity.

```
@Module
abstract class ActivityBuilder {
    @ContributesAndroidInjector
    internal abstract fun bindMainActivity(): MainActivity
}
```

The `AppModule` class will provide the various dependencies required by the application. For our example, it will just provide an instance of `Context` and `UserService`.

```
@Module
open class AppModule(val application: Application) {
    @Provides
    @Singleton
    internal open fun provideContext(): Context {
        return application
    }
@Provides
    @Singleton
    internal open fun provideUserService(context: Context): UserService {
        return UserService(context)
    }
}
```

The `AppComponent` class lets you build the object graph for the application.

```
@Singleton
@Component(modules = [(AndroidSupportInjectionModule::class), (AppModule::class), (ActivityBuilder::class)])
interface AppComponent {
@Component.Builder
    interface Builder {
        fun appModule(appModule: AppModule): Builder
        fun build(): AppComponent
    }
fun inject(application: ExamplesApplication)
}
```

Create a method that returns already built component, then inject this component into `onCreate()`.

```
open class ExamplesApplication : Application(), HasActivityInjector {
    @Inject lateinit var dispatchingActivityInjector: DispatchingAndroidInjector<Activity>
override fun onCreate() {
        super.onCreate()
        initAppComponent().inject(this)
    }
open fun initAppComponent(): AppComponent {
        return DaggerAppComponent
                .builder()
                .appModule(AppModule(this))
                .build()
    }
override fun activityInjector(): DispatchingAndroidInjector<Activity>? {
        return dispatchingActivityInjector
    }
}
```

### Setting up Dagger in the Test Application

In order to mock responses from the server, we need to create a new `Application` class that extends the above class.

```
class TestExamplesApplication : ExamplesApplication() {
override fun initAppComponent(): AppComponent {
        return DaggerAppComponent.builder()
                .appModule(MockApplicationModule(this))
                .build()
    }
@Module
    private inner class MockApplicationModule internal constructor(application: Application) : AppModule(application) {
        override fun provideUserService(context: Context): UserService {
            val mock = Mockito.mock(UserService::class.java)
            `when`(mock!!.displayUserName("Test")).thenReturn("Hello Test!")
            return mock
        }
    }
}
```

As you can see in the example above we used Mockito to mock `UserService` and assume results. We still need a new runner that will point to new application class with overwritten data.

```
class MockTestRunner : AndroidJUnitRunner() {
override fun onCreate(arguments: Bundle) {
        StrictMode.setThreadPolicy(StrictMode.ThreadPolicy.Builder().permitAll().build())
        super.onCreate(arguments)
    }
@Throws(InstantiationException::class, IllegalAccessException::class, ClassNotFoundException::class)
    override fun newApplication(cl: ClassLoader, className: String, context: Context): Application {
        return super.newApplication(cl, TestExamplesApplication::class.java.name, context)
    }
}
```

Next, you need to update the `build.gradle` file to use the `MockTestRunner`.

```
android {
   ...
defaultConfig {
        ...
        testInstrumentationRunner ".MockTestRunner"
    }
}
```

### Running the Test

All tests with new `TestExamplesApplication` and `MockTestRunner` should be added at `androidTest` package. This implementation makes the tests fully independent from the server and gives the ability to manipulate responses. With the above setup in place, our test class won't change at all. When the test is run, the app uses `TestExamplesApplication` instead of `ExamplesApplication` and thus a mocked instance of `UserService` gets used.

```
@RunWith(AndroidJUnit4::class)
class MainActivityTest {
    @Rule @JvmField var activityActivityTestRule = ActivityTestRule(MainActivity::class.java)
@Test
    fun setUserName() {
        onView(withId(R.id.name_field)).perform(typeText("Test"))
        onView(withId(R.id.set_user_name)).perform(click())
        onView(withText("Hello Test!")).check(matches(isDisplayed()))
    }
}
```

The test runs successfully when you click on the green play button in the IDE.

![](/post/img/2018-05-20_how-to-improve-coverage-for-your-android-app-using-mockito-and-espresso-part-2_6.png#layoutTextWidth)

That’s it, you have successfully setup Dagger and ran tests using Espresso and Mockito.

### Conclusion

The article highlights that the most important aspect of improving code coverage is to write testable code. Frameworks like Espresso and Mockito provide easy to use APIs which makes writing tests for various scenarios easier. Tests should be run in isolation so mocking the dependencies gives us an opportunity to ensure that objects perform the actions that are expected from them. There are a variety of Android testing tools available, and as the ecosystem matures, the process of setting up a testable environment and writing tests will become easier. Writing testable code requires some discipline, concentration, and extra effort. As a reward, we’ll end up with clean, easy-to-maintain, loosely coupled, and reusable APIs, that won’t damage developers’ brains when they try to understand it. The complete [source code](https://github.com/maskaravivek/AndroidTestingExamples) for the examples used in this article is available on Github. Feel free to take a look at it.

Check out this article to see a few examples of working with `RecyclerView.`

[Working with Recycler Views in Espresso Tests](/working-with-recycler-views-in-espresso-tests-6da21495182c "https://medium.com/@maskaravivek/working-with-recycler-views-in-espresso-tests-6da21495182c")

You can buy me a coffee if this post really helped you learn something or fix a nagging issue!

* * *
Written on May 20, 2018 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/how-to-improve-coverage-for-your-android-app-using-mockito-and-espresso-part-2-f4c2ac57bce8)
