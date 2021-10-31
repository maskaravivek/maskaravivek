---
title: "Querying Wikimedia Commons DB to get User Statistics"
author: "Vivek Maskara"
date: 2018-12-02T07:54:24.643Z
lastmod: 2021-10-29T21:24:07-07:00

description: ""

subtitle: ""

categories: [Open Source]

tags:
 - Open Source
 - Wikimedia Commons

image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2018-12-02_querying-wikimedia-commons-db-to-get-user-statistics_0.jpeg"
 - "/post/img/2018-12-02_querying-wikimedia-commons-db-to-get-user-statistics_1.png"


aliases:
- "/querying-wikimedia-commons-db-to-get-user-statistics-c155cf47a3db"

---

![](/post/img/2018-12-02_querying-wikimedia-commons-db-to-get-user-statistics_0.jpeg#layoutTextWidth)

[Wikimedia Commons](https://commons.wikimedia.org/wiki/Main_Page) is an online repository of free-use images, sounds, and other media files. It is a project of the Wikimedia Foundation.

If you are a Commons user or need to get statistics for a particular user, then you can query the `commonswiki` DB to get the results. In this post, I will introduce you to a few SQL queries to get various user statistics.

### Get Count of File Uploads

You can check the number of file uploads by a user using the following query:

```
use commonswiki_p;
select
  count(*)
from
  logging_userindex
where
  log_type = "upload"
  and log_user =(
    select
      user_id
    from
      user
    where
      user_name = "Maskaravivek"
  );
```

You can run the query on Quarry.

[Get Number of Uploads by a user - Quarry](https://quarry.wmflabs.org/query/31757 "https://quarry.wmflabs.org/query/31757")

### Get Count of Deleted file uploads

Commons has a strict [deletion policy](https://commons.wikimedia.org/wiki/Commons:Deletion_policy) and the community can [nominate a picture for deletion](https://commons.wikimedia.org/wiki/Commons:Deletion_requests) if your upload violates any of the policies.

You can use the following query to get the count of deleted file uploads.

```
select count(*) from commonswiki_p.filearchive_userindex where fa_user_text="Maskaravivek"
```

`commonswiki_p.filearchive_userindex` has an index on `username` and querying it is much faster than querying the original table ie. `commonswiki_p.filearchive` .

You can run the query on Quarry.

[Get deleted file uploads on Commons - Quarry](https://quarry.wmflabs.org/query/31719 "https://quarry.wmflabs.org/query/31719")

### Get Count of Articles Using Images

The whole purpose of the Commons image repository is to let Wikipedia articles use the images.

The query below gives the count of articles using images from a particular user.

```
select
  count(*) as articlesUsing
from
  commonswiki_p.globalimagelinks
where
  gil_to in (
    select
      log_title
    from
      commonswiki_p.logging_userindex
    where
      log_type = "upload"
      and log_user =(
        select
          user_id
        from
          commonswiki_p.user
        where
          user_name = "Maskaravivek"
      )
  );
```

You can run the query on Quarry.

[Query to get the count of articles using images - Quarry](https://quarry.wmflabs.org/query/31752 "https://quarry.wmflabs.org/query/31752")

### Get Count of Unique Images USed

The query below gives the count of articles using images from a particular user.

```
select
  count(distinct gil_to) as uniqueUsed
from
  commonswiki_p.globalimagelinks
where
  gil_to in (
    select
      log_title
    from
      commonswiki_p.logging_userindex
    where
      log_type = "upload"
      and log_user =(
        select
          user_id
        from
          commonswiki_p.user
        where
          user_name = "Maskaravivek"
      )
  );
```

You can run the query on Quarry.

[Get count of distinct images used - Quarry](https://quarry.wmflabs.org/query/31754 "https://quarry.wmflabs.org/query/31754")

### Get Count of Images Edited by Someone else

The pictures that you upload to commons can be edited by some other user as well. The following query can be used to get the count of such images.

```
use commonswiki_p;

select
  count(*)
from
  revision
where
  rev_page in (
    select
      log_page
    from
      logging_userindex
    where
      log_type = "upload"
      and log_user =(
        select
          user_id
        from
          user
        where
          user_name = "Maskaravivek"
      )
  )
  and rev_user !=(
    select
      user_id
    from
      user
    where
      user_name = "Maskaravivek"
  )
group by
  rev_page
having
  count(*) > 1
```

You can try the query on Quarry.

[Commons Images Edited by Someone else - Quarry](https://quarry.wmflabs.org/query/31755 "https://quarry.wmflabs.org/query/31755")

### Get Number of Thanks received

Users can [express thanks](https://en.wikipedia.org/wiki/Wikipedia:Expressing_thanks) to other users using Commons.

The following query can be used to get the count of the number of thanks received by a user.

```
use commonswiki_p;

select count(*) from logging_logindex where log_type="thanks" and log_title="Maskaravivek";
```

[Get number of thanks received on Wikimedia Commons - Quarry](https://quarry.wmflabs.org/query/31756 "https://quarry.wmflabs.org/query/31756")

The [Commons Android app](https://play.google.com/store/apps/details?id=fr.free.nrw.commons) shows these statistics in a very nice interface.

![](/post/img/2018-12-02_querying-wikimedia-commons-db-to-get-user-statistics_1.png#layoutTextWidth)

Make sure you give this post **50 claps**and **follow**meif you enjoyed this post and want to see more!

* * *
Written on December 2, 2018 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/querying-wikimedia-commons-db-to-get-user-statistics-c155cf47a3db)
