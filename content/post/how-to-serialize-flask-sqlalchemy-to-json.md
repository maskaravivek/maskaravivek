---
title: "How to Serialize Flask SQLAlchemy to JSON"
author: "Vivek Maskara"
date: 2020-05-10T17:29:09.922Z
lastmod: 2021-10-29T21:25:28-07:00

description: ""

subtitle: ""

categories: [Python Flask]

tags:
 - Sqlalchemy
 - Flask
 - Python
 - JSON

image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2020-05-10_how-to-serialize-flask-sqlalchemy-to-json_0.png"
 - "/post/img/2020-05-10_how-to-serialize-flask-sqlalchemy-to-json_1.png"


aliases:
- "/how-to-serialize-flask-sqlalchemy-to-json-2ed736b67c92"

---

![](/post/img/2020-05-10_how-to-serialize-flask-sqlalchemy-to-json_0.png#layoutTextWidth)

Recently I was involved in developing a [PPE response network](https://pperesponsenetwork.asu.edu/) built by The Luminosity Lab at Arizona State University. Its backend was built using Flask. This was my first time working on a Flask application and I was surprised that it is not trivial to elegantly serialize the database model classes to JSON. There are multiple approaches that can be used to serialize the classes. In this post, I will introduce you to the approach that I liked the most.

### Using SQLAlchemy-serializer

The SQLAlchemy-serializer library makes serializing objects to JSON quite easy and it has got quite a few customizations that let you handle all of your custom use cases.

[n0nSmoker/SQLAlchemy-serializer](https://github.com/n0nSmoker/SQLAlchemy-serializer "https://github.com/n0nSmoker/SQLAlchemy-serializer")

#### Install the library

Install the library using `pip`.

```
pip install SQLAlchemy-serializer
```

#### Define the Model Class

Define your database’s model class using `SQLAlchemy`.

```
class Login(db.Model, CustomSerializerMixin):
    __tablename__ = 'login'

    serialize_only = ('id', 'email_id', 'role_type', 'users.id')
    
    serialize_rules = ('-merchants')

    id = db.Column(UUID(as_uuid=True), primary_key=True, unique=True)
    email_id = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role_type = db.Column(db.Enum(Role))
    users = db.relationship('Users', backref='login', uselist=False)
    merchants = db.relationship('Merchants', backref='login', uselist=False)
```

**Define a Custom Serializer**

If your database table uses classes like `UUID` , `Geometry` etc., you will need to define a custom serializer for it.

```
class CustomSerializerMixin(SerializerMixin):
    serialize_types = (
        (UUID, lambda x: str(x)),
    )
```

Check out the [documentation](https://github.com/n0nSmoker/SQLAlchemy-serializer#custom-types) to see what data types are natively supported by the library.

**Define fields to serialize**

You can specify the list of fields you want to serialize by using the `serialize_only` field in your model class.

```
serialize_only = ('id', 'email_id', 'role_type', 'users.id')
```

Note: In this example, we have a `users` relationship defined and we want to serialize just the `id` field of `users`.

**Define fields to exclude**

You can exclude the whole relationship object by using a minus(`-`) symbol before the relationship name.

```
serialize_rules = ('-merchants')
```

In this example, we want to exclude all the fields of the `merchants` field when serializing the JSON.

#### Avoid circular dependency

If you have a relationship and foreign key defined in your models, you could run into a circular dependency issue. Suppose you have the following model classes defined:

```
class Login(db.Model, CustomSerializerMixin):
  __tablename__ = 'login'

  serialize_only = ('id', 'email_id', 'role_type', 'users')

  id = db.Column(UUID(as_uuid=True), primary_key=True, unique=True)
  email_id = db.Column(db.String(120), unique=True, nullable=False)
  password = db.Column(db.String(200), nullable=False)
  role_type = db.Column(db.Enum(Role))
  users = db.relationship('Users', backref='login', uselist=False)
  merchants = db.relationship('Merchants', backref='login', uselist=False)
    
class Users(db.Model, CustomSerializerMixin):
  __tablename__ = 'users'

  id = db.Column(UUID(as_uuid=True), primary_key=True)
  first_name = db.Column(db.String(120), nullable=False)
  middle_name = db.Column(db.String(120), nullable=True)
  last_name = db.Column(db.String(120), nullable=False)
  phone_extension = db.Column(db.String(10), nullable=False)
  phone_number = db.Column(db.String(20), nullable=False)
  login_id = db.Column(UUID(as_uuid=True), db.ForeignKey('login.id'), nullable=False, unique=True)
```

Notice that the login class has `users` added in its `serialize_only` field. With this setup you will face a maximum recursion depth exceeded exception. The serializer will find an instance of the same class among the model’s relationships. To avoid this you need to define where the serialization should stop:

```
serialize_rules = ('-users.login.users',)
```

#### Serialize JSON

With the serializer mixin in place, its very simple to serialize your object to JSON.

```
login.to_dict()
```

You can buy me a coffee if this post really helped you learn something or fix a nagging issue!

* * *
Written on May 10, 2020 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/how-to-serialize-flask-sqlalchemy-to-json-2ed736b67c92)
