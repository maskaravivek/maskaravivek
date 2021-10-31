---
title: "Automate API testing using Postman Collection"
author: "Vivek Maskara"
date: 2020-04-29T03:03:14.309Z
lastmod: 2021-10-29T21:25:21-07:00

description: ""

subtitle: ""

categories: [Automation Testing]

tags:
 - Postman
 - API testing
 - Automation


image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2020-04-29_automate-api-testing-using-postman-collection_0.jpeg"


aliases:
- "/automate-api-testing-using-postman-collection-eb2e5f2f59e8"

---

![](/post/img/2020-04-29_automate-api-testing-using-postman-collection_0.jpeg#layoutTextWidth)

[Postman](https://www.postman.com/) is a very popular tool used by a lot of developers to collaborate for API development. It is also a very powerful tool and can be used for end to end automated API testing.

In this post, I will show you various examples of writing pre-request and test scripts in Postman. I found these methods very helpful while writing automated test cases for a [project](https://pperesponsenetwork.asu.edu/) that I worked on recently.

### Pre-Request Scripts

You can define pre-request scripts which are executed before the API request is made. These scripts could be useful for generating random data and using it in the API requests.

#### Generating random Email ID and Password

While writing automated API tests, you might need to create a new user every time your collection is executed. For eg. to test a **create user**API the email ID needs to be unique or else your API might throw an error. This code snippet generates a random email ID and password and sets it to the environment variables.

```
const uuid = Math.random().toString(36).substring(2, 15) +
        Math.random().toString(36).substring(2, 15);
const email = "user_" + uuid + "@example.com";
const password = "pass_" + uuid;
pm.environment.set("EMAIL_ID", email);
pm.environment.set("PASSWORD", password);
```

Now, you can use these environment variables in your API requests using:

```
{
  "email": "{{EMAIL_ID}}",
  "password": "{{PASSWORD}}"
}
```

Similarly, you can utilize the pre-request scripts for generating other random data or just for setting environment variables.

#### Setting an environment variable

```
pm.environment.set("variable_key", "variable_value");
```

#### Setting a global variable

```
pm.globals.set("variable_key", "variable_value");
```

#### Making an API request

You can also make an API request in your pre-request script to fetch some data that might be required by your current API. For eg., you might need to fetch the user’s ID and use it in your request.

```
pm.sendRequest("https://example.com/get", function (err, response) {
  // set env var
  pm.environment.set("USER_ID", response.json().id);
});
```

### Test Scripts

Test scripts can be used for testing the validity of the API response.

#### Status Check

The most basic test for any API would be status check:

```
pm.test("Status code is 200", function () { pm.response.to.have.status(200); });
```

#### Check API response

The status check is a very basic test and in most scenarios, you might want to write more advanced tests that validate your API’s response.

**Checking for null values**

You can easily check if any of the attributes in the JSON response is undefined or not.

```
pm.test("Response is correct", function () {
    pm.expect(pm.response.json().auth_token).to.not.be.undefined;
});
```

**Check for Response keys**

In some scenarios, you might want to ensure that the API is not leaking any extra data and only the required fields are being returned. The following snippet will check

```
pm.test("Response is correct", function () {
  pm.expect(pm.response.json().user_data).to.have.all.keys('email_id', 'id', 'role_type')
});
```

**Check for a non-empty array**

You might want to check if your API response is not returning an empty array.

```
pm.test("Response is correct", function () {
    pm.expect(pm.response.json()).to.be.an('array').that.is.not.empty;
});
```

**Match response with environment variable**

You can use this snippet to check if the response attribute matches some environment variables.

```
pm.test("Response is correct", function () {
  pm.expect(pm.response.json().user_id).to.equal(pm.environment.get("USER_ID"))
});
```

**Test API response time**

You can also write a script to ensure that your API is responding below a certain time threshold.

```
pm.test("Response time check", function () {
    pm.expect(pm.response.responseTime).to.be.below(200);
});
```

#### **Set an Environment Variable**

If you need to set an environment variable based on the API response you can do it as follows:

```
pm.environment.set("USER_ID", "pm.response.json().id");
```

I hope you would find these examples useful. To learn about how to run a postman collection refer to this doc:

[Starting a collection run](https://learning.postman.com/docs/postman/collection-runs/starting-a-collection-run/ "https://learning.postman.com/docs/postman/collection-runs/starting-a-collection-run/")

* * *
Written on April 29, 2020 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/automate-api-testing-using-postman-collection-eb2e5f2f59e8)
