---
title: "How to Import Data to Neo4J from CSV files"
author: "Vivek Maskara"
date: 2020-06-26T19:15:17.142Z
lastmod: 2021-10-29T21:26:08-07:00

description: ""

subtitle: ""

categories: [Neo4J, Databases]

tags:
 - Graph Database
 - Neo4J


image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2020-06-26_how-to-import-data-to-neo4j-from-csv-files_0.png"
 - "/post/img/2020-06-26_how-to-import-data-to-neo4j-from-csv-files_1.png"
 - "/post/img/2020-06-26_how-to-import-data-to-neo4j-from-csv-files_2.png"
 - "/post/img/2020-06-26_how-to-import-data-to-neo4j-from-csv-files_3.png"
 - "/post/img/2020-06-26_how-to-import-data-to-neo4j-from-csv-files_4.png"
 - "/post/img/2020-06-26_how-to-import-data-to-neo4j-from-csv-files_5.png"


aliases:
- "/how-to-import-data-to-neo4j-from-csv-files-9c95905023bb"

---

![](/post/img/2020-06-26_how-to-import-data-to-neo4j-from-csv-files_0.png#layoutTextWidth)

In this post, we will see how to import data to the Neo4J database from CSV files.

Recently, I worked with Graph databases for the first time and was really amazed by the capabilities it offers. Initially, I struggled to find a good resource that guides beginners to import CSV data to the Neo4J database. So I thought it might be useful to share the steps using a real-world example.

### Prerequisite

Download and install Neo4J on your machine.

[Neo4j Desktop Download - Launch and Manage Neo4j Databases](https://neo4j.com/download/ "https://neo4j.com/download/")

Next, start Neo4J and create a new “Local Graph” named “user_sample”.

![](/post/img/2020-06-26_how-to-import-data-to-neo4j-from-csv-files_1.png#layoutTextWidth)

### Preparing the CSV

Let us assume that you have the following CSV file with User data. I have chosen an example with different types of data values to explore various common scenarios.

![](/post/img/2020-06-26_how-to-import-data-to-neo4j-from-csv-files_2.png#layoutTextWidth)

If your data is stored in an Excel sheet saved on Google Drive then you can directly import the data by first publishing the file on the web. You can go to **File > Publish to Web**to get a link.

![](/post/img/2020-06-26_how-to-import-data-to-neo4j-from-csv-files_3.png#layoutTextWidth)

**Note:** If you try to use the doc link directly, it won’t work.

- If you have a link to a CSV hosted elsewhere, you can use that directly.
- If you have a locally stored CSV file, you can follow the instructions [here](https://neo4j.com/developer/desktop-csv-import/#about-desktop-import) to copy the CSV to the appropriate path and then use the file path.

Let’s see how can you import this data to Neo4J.

### Importing Data from CSV

You can choose to structure your nodes and relationships in any way that suits your application but for this example, I have decided to create separate nodes for address, phone number, email, and user.

Let us start off by defining uniqueness constrains for these nodes.

```
CREATE CONSTRAINT ON (c:Address) ASSERT c.address IS UNIQUE;
CREATE CONSTRAINT ON (c:PhoneNumber) ASSERT c.phone_number IS UNIQUE;
CREATE CONSTRAINT ON (c:Email) ASSERT c.email IS UNIQUE;
CREATE CONSTRAINT ON (c:User) ASSERT c.id IS UNIQUE;
```

This will ensure that a duplicate node is not created for the same data.

Finally, let us import the CSV data. We will use the `LOAD CSV` Cypher statement to import data from CSV. Start the Neo4J graph DB and execute the query in the Cypher query window.

```
LOAD CSV WITH HEADERS FROM "https://docs.google.com/spreadsheets/d/e/2PACX-1vRWR2ZZy7YL4s0xSc6dJK1nA4GtVD93yCco35B5ghD6jdLvOUC--f6u_AmtA9Ob1NJn9RrLGjdR8Q04/pub?gid=0&single=true&output=csv" AS line
WITH line, SPLIT(replace(replace(replace(line.wallets, '"', ''), '[', ''), ']', ''), ',') AS wallets
MERGE (user: User {user_id:line.user_id, name:line.name, created_date:date(line.created_at), citizenship_country: line.country})
MERGE (phone_number:PhoneNumber {phone_number: line.phone_number})
MERGE (email:Email {email:line.email})
MERGE (address: Address {address:line.address})
MERGE (user)-[:MAILING_ADDRESS]->(address)
MERGE (user)-[:CONTACT_PHONE_NUMBER]->(phone_number)
MERGE (user)-[:CONTACT_EMAIL_ID]->(email)
FOREACH (wallet IN wallets| MERGE (w:Wallet {wallet_id: wallet}))
```

Let us discuss a few things about the above query:

- `LOAD CSV WITH HEADERS FROM <link> WITH line` reads the CSV file line by line taking the first line as the header.
- `created_at` is imported as a date data type. See [docs](https://neo4j.com/developer/dates-datetimes-durations/) for more date-time formatting options.
- We are using `MERGE` instead of `CREATE` so that if a node already exists then it's updated instead of creating a new node.
- We are using the `SPLIT` function to split the wallet IDs to `wallets` variable.
- Finally, we use `FOREACH` to iterate all `wallets` and create `Wallet` node using it.

Now, that you have imported the data, you can visualize it in Neo4J.

![](/post/img/2020-06-26_how-to-import-data-to-neo4j-from-csv-files_4.png#layoutTextWidth)

That’s it for this post. I hope you would find this post useful.

You can buy me a coffee if this post really helped you learn something or fix a nagging issue!

* * *
Written on June 26, 2020 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/how-to-import-data-to-neo4j-from-csv-files-9c95905023bb)
