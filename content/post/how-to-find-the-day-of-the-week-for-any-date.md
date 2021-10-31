---
title: "How to find the day of the week for any Date?"
author: "Vivek Maskara"
date: 2020-05-20T09:25:50.455Z
lastmod: 2021-10-29T21:25:34-07:00

description: ""

subtitle: ""

categories: [Coding]

tags:
 - Coding

aliases:
- "/how-to-find-the-day-of-the-week-for-any-date-a8d6d0adc7c1"

---

In this post, I will show you how you can calculate the day of the week for any date using Zeller’s formula.

The formula was proposed by a Reverend Zeller.

[Zeller's congruence](https://en.wikipedia.org/wiki/Zeller%27s_congruence "https://en.wikipedia.org/wiki/Zeller%27s_congruence")

Here’s the formula.

```
f = k + [(13*m-1)/5] + D + [D/4] + [C/4] - 2*C.
```

**Note** [x] stands for the greatest integer smaller than or equal to x. For eg. [5.69] = 5.

Let us suppose we have to calculate the day of the week for 31st August 2019.

- k is the day of the month. For the given example k = 31.
- m is the month number when starting counting from March. March is marked as 1, April as 2, and so on. Therefore January is marked as 1 and February is 2. For the given example m = 6.
- D is the last two digits of the year. If the month is January or February, then D is subtracted by 1. For the given example D = 19 as the month is August. For 31st January 2019, D = 18.
- C is the first two digits of the year. For the given example C = 20.

Now, let us calculate the value of `f` for the above date.

```
f = k + [(13*m-1)/5] + D + [D/4] + [C/4] - 2*C
= 31 + [(13*6-1)/5] + 19 + [19/4] + [20/4] - 2*20
= 31 + [15.4] + 19 + [4.75] + [5] - 40
= 31 + 15 + 19 + 4 + 5 - 40
= 34.
```

Now that we know the value of `f` let us divide it by 7 to calculate the remainder. Note, that if the remainder is negative, we add 7 to make it positive.

```
34 % 7 // finding remainder
= 6
```

A remainder of 0 denotes Sunday, 1 means Monday, etc. So for our example, 31st August 2019 would be **Saturday**.

This problem is available on LeetCode.

[Day of the Week - LeetCode](https://leetcode.com/problems/day-of-the-week/ "https://leetcode.com/problems/day-of-the-week/")

Here’s the code to implement the same in Java.

```
public String dayOfTheWeek(int day, int month, int year) {
    String[] days = new String[]{"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"};
    int k = day;
    int m = month < 3 ? month + 10 : month - 2;
    int d = month < 3 ? (year % 100) - 1 : (year % 100);
    int c = year / 100;
    double f = k + Math.floor((13.0 * m - 1.0) / 5.0) + d + Math.floor(d / 4.0) + Math.floor(c / 4.0) - 2.0 * c;
    int rem = (int) f % 7;
    if (rem < 0) {
        rem = 7 + rem;
    }
    return days[rem];
}
```

* * *
Written on May 20, 2020 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/how-to-find-the-day-of-the-week-for-any-date-a8d6d0adc7c1)
