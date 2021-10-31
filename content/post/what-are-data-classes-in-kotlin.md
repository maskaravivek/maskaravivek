---
title: "What are Data Classes in Kotlin?"
author: "Vivek Maskara"
date: 2018-03-04T13:52:27.778Z
lastmod: 2021-10-29T21:23:29-07:00

description: ""

subtitle: ""


categories: [Android]

tags:
 - Android
 - Kotlin

aliases:
- "/what-are-data-classes-in-kotlin-d58f670cc64"

---

Recently, I started Reading Kotlin In Action by Dmitry Jemerov and learned about Data classes in Kotlin. It's one of the best books out there to get started with Kotlin. You can buy the [paperback edition of the book](https://amzn.to/3oRQAYa) from Amazon.

A number of methods need to present in many classes of Java and are usually implemented manually, such as `equals`, `hashCode` and `toString`. IDEs such as IntelliJ can automatically generate these methods saving the manual effort but still, the codebase contains the boilerplate code. Kotlin compiler can perform the mechanical code-generation behind the scenes, without cluttering your code files.

Let's take an example of the `User` class which contains the `userName` and `age `properties.

```
class User(val userName: String, val age: Int)
```

The above class contains can be used as:

```
User("Vivek", 25)
```

As in Java, Kolin classes also have several methods that you may want to override. Let's have a look at these methods,

#### String Representation: ToString()

All classes in Kolin, provide a way to get a string representation of the class’s object and are primarily used for debugging and logging. Let's print the default string representation of an object.

```
open fun printUser() {
    var user = User("Vivek", 25)
    Log.d("UserService", "To string is: " + user.toString())
}
```

The above class gives the following output.

```
com.maskaravivek.testingexamples.model.User@e30a606
```

To change this you need to override the default `toString` method.

```
class User(val userName: String, val age: Int) {
    override fun toString(): String = "User(name=$userName, age=$age)"
}
```

As expected, you will get the following output using the same `printUser` method.

```
User(name=Vivek, age=25)
```

#### Equality of Object: Equals()

As all the computations take place outside the class and the `User` class contains just the data. Suppose you want the objects to be considered equal if they contain the same data. Let the function method check for the equality of `User` objects.

```
open fun areUsersEqual() {
    var user1 = User("Vivek", 25)
    var user2 = User("Vivek", 25)
    Log.d("UserService", "Are users equal: " + (user1 == user2))
}
```

The function above returns `false`. In Kotlin, the `==` operator is the default way to compare two objects. It compares their values by calling `equals` under the hood. Thus if `equals` is overridden in your classs, you can compare its instances using `==`. For reference comparison, you can use the `===` operator.

Let's implement the `equals` method in `User`. Here’s the modified `User` class.

```
class User(val userName: String, val age: Int) {
    override fun toString(): String = "User(name=$userName, age=$age)"
    override fun equals(other: Any?): Boolean {
        if(other == null || other !is User) {
            return false
        }
        return userName == other.userName && age == other.age
    }
}
```

Executing the same method, `areUsersEqual` again prints true. Now lets try to execute the following method:

```
open fun checkContains() {
    val users = hashSetOf(User("Vivek", 25))
    Log.d("UserService", "Is User contained: " + (users.contains(User("Vivek", 25))))
}
```

This method prints false. The reason is that the `User` class is missing the `hashCode` method. So it violates the general `hashCode` contract. If two objects are equal, they must have the same hash code. The `users` set is a `HashSet`. Values in a `HashSet` are compared in an optimized way: at first, their hash codes are compared, and then only if they are equal, the actual values are compared. For the above examples `hashCode` is different for a different instance of `User`. To fix that you can add the implementation of `hashCode` to the class.

```
class User(val userName: String, val age: Int) {
    override fun toString(): String = "User(name=$userName, age=$age)"
    override fun equals(other: Any?): Boolean {
        if(other == null || other !is User) {
            return false
        }
        return userName == other.userName && age == other.age
    }

    override fun hashCode(): Int {
        return userName.hashCode() * 10 + age
    }
}
```

Basically, if you want your class to be a convenient holder for your data, you need to override these methods `equals`, `hashCode` and `toString`. Simply adding the `data` modifier to your class generates the necessary methods automatically for you.

**Note**: The properties that aren’t declared in the primary constructor don’t take part in the equality checks and hash code calculation.

Moreover, data classes generate a few other methods for you.

#### Immutability: The Copy() method

For data classes, it strongly recommended that you use only read-only properties, making the instances of the data class `immutable`. Immutable objects are much easier to reason about, especially in multithreaded code. Kotlin compiler generates a method that allows you to `copy` the instances of your class, changing the values of some properties. Here’s how the copy method would look like if you implemented it manually.

```
fun copy(userName: String = this.userName, age: Int = this.age) = User(userName, age)
```

And here’s how the copy method can be used:

```
open fun checkCopy() {
    val user = User("Vivek", 25)
    val userCopy = user.copy(age = 26)
    Log.d("UserService", "user copy is: " + userCopy.toString())
}
```

### Magic of Data Class

Our `User` class looks like the following without the `data` modifier.

```
class User(val userName: String, val age: Int) {
    override fun toString(): String = "User(name=$userName, age=$age)"
    
    override fun equals(other: Any?): Boolean {
        if(other == null || other !is User) {
            return false
        }
        return userName == other.userName && age == other.age
    }

    override fun hashCode(): Int {
        return userName.hashCode() * 10 + age
    }

    fun copy(userName: String = this.userName, age: Int = this.age) = User(userName, age)
}
```

Let's execute the following functions on it:

```
printUser()
areUsersEqual()
checkContains()
checkCopy()
```

As expected, you get the following output:

```
To string is: User(name=Vivek, age=25)
Are users equal: true
Is User contained: true
user copy is: User(name=Vivek, age=26)
```

Now let's use the `data` modifier in the class.

```
data class User(val userName: String, val age: Int)
```

And run the above methods again. Magic happens and you get the same output.

```
To string is: User(userName=Vivek, age=25)
Are users equal: true
Is User contained: true
user copy is: User(userName=Vivek, age=26)
```

You can see how the `data` modifier reduces boilerplate and makes value-object classes more convenient to use.

* * *
Written on March 4, 2018 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/what-are-data-classes-in-kotlin-d58f670cc64)
