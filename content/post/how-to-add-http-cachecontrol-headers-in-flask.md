---
title: "How to add HTTP Cache-Control Headers in Flask"
author: "Vivek Maskara"
date: 2020-04-22T00:39:53.483Z
lastmod: 2021-10-29T21:25:16-07:00

description: ""

subtitle: ""

categories: [Python Flask]

tags:
 - Python
 - Caching
 - Flask

image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2020-04-22_how-to-add-http-cachecontrol-headers-in-flask_0.png"
 - "/post/img/2020-04-22_how-to-add-http-cachecontrol-headers-in-flask_1.png"
 - "/post/img/2020-04-22_how-to-add-http-cachecontrol-headers-in-flask_2.png"


aliases:
- "/how-to-add-http-cache-control-headers-in-flask-34659ba1efc0"

---

![](/post/img/2020-04-22_how-to-add-http-cachecontrol-headers-in-flask_0.png#layoutTextWidth)

Using caching can vastly improve the performance of your applications. There are different types of caching options available while developing Web applications but HTTP caching is one of the simplest ways to improve the performance of your application.

As described in [MDN web docs](https://developer.mozilla.org/en-US/docs/Web/HTTP/Caching):

> The [`Cache-Control`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Cache-Control) HTTP/1.1 general-header field is used to specify directives for caching mechanisms in both requests and responses.

In this short post, I will show you how you could use [decorators in Flask](https://flask.palletsprojects.com/en/1.1.x/patterns/viewdecorators/) to cleanly add HTTP cache-control headers to your APIs.

Firstly, add a decorator method `docache` in your `utils.py` file. The decorator adds the cache headers to the Flask response. The method takes two parameters:

- **minutes:**To define the age and expiry of the cache
- **content_type:**To define the content type of the response.

```
from datetime import datetime, date, timedelta
from functools import wraps
from flask import Response

def docache(minutes=5, content_type='application/json; charset=utf-8'):
    """ Flask decorator that allow to set Expire and Cache headers. """
    def fwrap(f):
        @wraps(f)
        def wrapped_f(*args, **kwargs):
            r = f(*args, **kwargs)
            then = datetime.now() + timedelta(minutes=minutes)
            rsp = Response(r, content_type=content_type)
            rsp.headers.add('Expires', then.strftime("%a, %d %b %Y %H:%M:%S GMT"))
            rsp.headers.add('Cache-Control', 'public,max-age=%d' % int(60 * minutes))
            return rsp
        return wrapped_f
    return fwrap
```

Using decorators makes life easier and now you can add HTTP cache control headers in any of your API responses with just a single line of code. You can simply annotate the method with `@docache` to add the headers.

```
from flask import json
from .utils import docache

@api.route('/summary')
@docache(minutes=5, content_type='application/json')
def get_summary():
    return json.dumps(range({"count": 5}))
```

That’s it. Now if your CDN is configured properly, you would start seeing cache hits for your API calls.

![](/post/img/2020-04-22_how-to-add-http-cachecontrol-headers-in-flask_1.png#layoutTextWidth)

You can buy me a coffee if this post really helped you learn something or fix a nagging issue!

* * *
Written on April 22, 2020 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/how-to-add-http-cache-control-headers-in-flask-34659ba1efc0)
