---
title: "Introduction to Async-Await in Javascript"
author: "Vivek Maskara"
date: 2018-09-30T09:33:07.633Z
lastmod: 2021-10-29T21:24:02-07:00

description: ""

subtitle: ""

categories: [Javascript]

tags:
 - Async-Await
 - Javascript

image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2018-09-30_introduction-to-asyncawait-in-javascript_0.jpeg"
 - "/post/img/2018-09-30_introduction-to-asyncawait-in-javascript_1.png"
 - "/post/img/2018-09-30_introduction-to-asyncawait-in-javascript_2.png"


aliases:
- "/introduction-to-async-await-in-javascript-3461285e92cc"

---

![](/post/img/2018-09-30_introduction-to-asyncawait-in-javascript_0.jpeg#layoutTextWidth)

Today I learned that Javascript has a special syntax for working with promises in a more comfortable fashion. It's the `async` and `await` which was originally introduced in C#. I am not an expert in Javascript but am quite excited to share the basics about `async` and `await`.

#### Async

`async` before a function means that the function always returns a promise. If the code has a non-promise return type then Javascript automatically wraps it into a resolved promise with that value.

Let's see how a normal promise looks in javascript.

```
var promise = new Promise(function(resolve, reject) {
    setTimeout(function() { 
        resolve('hello world');
    }, 300);
});

promise.then(function(value) { 
    console.log(value);
});

console.log(promise);
```

Here’s a simple example of a function with `async` keyword.

```
async function myFunc() {
    return "Hello world"
}

myFunc().then(data => console.log(data))
```

Executing the above function gives the following result.

![](/post/img/2018-09-30_introduction-to-asyncawait-in-javascript_1.png#layoutTextWidth)

Read more about Async function [here](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/async_function).

#### Await

`await` makes Javascript wait until that promise settles and returns its result. It can only be used inside an `async` function. The promise could either resolve or get rejected and accordingly the following happens:

- The execution of the `async` function continues once the promise is resolved and the value of the `await` expression is that of the fulfilled promise.
- If the promise is rejected, the `await` expression throws the rejected value.

Here’s an example of `await` in a function.

```
async function myFunc() {
  let promise = new Promise((resolve, reject) => {
    setTimeout(() => resolve("hello world!"), 1000)
  });

let result = await promise;

console.log(result);
}

myFunc();
```

This is how the output looks:

![](/post/img/2018-09-30_introduction-to-asyncawait-in-javascript_2.png#layoutTextWidth)

**Note:**You cannot use `await` in a normal function. It can be used only in an `async` function.

Read more about Await keyword [here](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/await).

* * *
Written on September 30, 2018 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/introduction-to-async-await-in-javascript-3461285e92cc)
