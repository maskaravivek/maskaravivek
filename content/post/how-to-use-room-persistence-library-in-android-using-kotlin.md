---
title: "How to use Room Persistence library in Android using Kotlin"
author: "Vivek Maskara"
date: 2020-06-09T20:42:38.932Z
lastmod: 2021-10-29T21:25:46-07:00

description: ""

subtitle: ""

categories: [Android]

tags:
 - Android
 - Room library
 - Kotlin

image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2020-06-09_how-to-use-room-persistence-library-in-android-using-kotlin_0.jpeg"


aliases:
- "/how-to-use-room-persistence-library-in-android-using-kotlin-746b5ad05b7f"

---

![](/post/img/2020-06-09_how-to-use-room-persistence-library-in-android-using-kotlin_0.jpeg#layoutTextWidth)

In this post, I will show you how to quickly add Room support for Room database in your app using Kotlin. There are quite a few articles on how to use Room but most of them are not up to date.

Earlier Room was part of the `android.arch.persistence` library but recently it was moved to `androidx` . I will walk you through the basic steps for setting up Room in your Android application.

#### **Add library dependencies**

Go ahead and add these dependencies in your `app/build.gradle` file. If you are using Kotlin, make sure to use `kapt` instead of `annotationProcessor` for room compiler.

```
implementation "androidx.room:room-runtime:$room_version"
kapt "androidx.room:room-compiler:$room_version" // For Kotlin use kapt instead of annotationProcessor

// optional - RxJava support for Room
implementation "androidx.room:room-rxjava2:$room_version"
```

#### Define the Entities

Next, define the tables for your room database. In Room world, tables are referred to as `Entity` . You can annotate any existing model class with the `@Entity` annotation to use it as an entity.

```
@Entity
data class User(
    @PrimaryKey(autoGenerate = true) val uid: Int,
    @ColumnInfo(name = "first_name") val firstName: String?,
    @ColumnInfo(name = "last_name") val lastName: String?
)
```

In the above example, we are defining the `User` entity.

#### Define the Dao

The Dao layer lets you query the database. You need to define an `interface` and annotate it with `@Dao` . You can either write raw SQL queries or use one of the predefined annotations for simple `Insert`, `Delete` or `Update`.

```
@Dao
interface UserDao {
    @Query("SELECT * FROM user")
    fun loadAll(): List<User>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    fun save(users: List<User>): Single<List<Long>>

    @Delete
    fun delete(user: User)
}
```

#### Define the Database

Next, you need to define an abstract class that extends the `RoomDatabase` . This class would give us the instances of the `Dao` that we defined earlier.

```
@Database(entities = arrayOf(User::class), version = 1)
abstract class AppDatabase : RoomDatabase() {
    abstract fun userDao(): UserDao
}
```

Note that you would need to annotate this class and pass a list of entities for the tables you want to have in your database. In this example, we are passing the `User` entity.

#### Init the Database

Now, you need to create an instance of the `AppDatabase` that you just defined. Its usually recommended to define it as a `Singleton` and use the same instance across the whole application.

```
fun initDb(): AppDatabase {
    return Room.databaseBuilder(
        applicationContext,
        AppDatabase::class.java, "user-db"
    ).build()
}
```

#### Use the Database

Now, that we have the whole database setup, we can go ahead and insert data into it. First, we will write a few helper functions to get a mock list of users. In an actual application, this would be replaced by business logic to fetch the data either from an API call or through other user actions.

We would be calling the `mockGetUsers` function later to get a list of 10 users.

```
fun mockGetUsers(): Observable<MutableList<User>> {
    val users = mutableListOf<User>()
    for (i in 0..9) {
        users.add(i, getUser())
    }
    return Observable.just(users)
}

fun getUser(): User {
    return User(0, getRandomString(5), getRandomString(8))
}

fun getRandomString(length: Int): String {
    val allowedChars = ('A'..'Z') + ('a'..'z')
    return (1..length)
        .map { allowedChars.random() }
        .joinToString("")
}
```

**Insert users to the Database**

Now, that we have a list of mocked users, we will go ahead and call the `fetchUsers` function to fetch a mocked list of users and save it to the database.

```
fun fetchUsers() {
    compositeDisposable.add(
        mockGetUsers()
            .subscribeOn(Schedulers.io())
            .subscribe(
                ::saveUsersToDb
            ) { error: Throwable ->
                // do nothing
            }
    )
}

/**
 * Saves the users to the DB
 */
private fun saveUsersToDb(users: List<User>) {
    compositeDisposable.add(
        userDao!!.save(users)
            .subscribeOn(Schedulers.io())
            .subscribe { longs: List<Long?>? ->
                //do nothing
            }
    )
}
```

Note that any DB operations should be performed on a background thread. In this example, we are using `Schedulers.io()` to get a new background thread.

**Fetch users from the database**

Similarly, its straight forward to fetch users from the database.

```
var users = userDao.loadAll()
```

The above function will fetch a list of all users from the database.

That’s it. The basic integration of the Room is quite easy to use. In my next article, I will talk about using Android Paged Lists to lazily load and display data in a `RecyclerView`.

* * *
Written on June 9, 2020 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/how-to-use-room-persistence-library-in-android-using-kotlin-746b5ad05b7f)
