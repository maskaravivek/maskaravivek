---
title: "Understanding the format of NDEF Messages"
author: "Vivek Maskara"
date: 2018-05-20T16:39:59.468Z
lastmod: 2021-10-29T21:23:57-07:00

description: ""

subtitle: ""

categories: [Android]

tags:
 - Android
 - NFC
 - NDEF

image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2018-05-20_understanding-the-format-of-ndef-messages_0.png"


aliases:
- "/understanding-the-format-of-ndef-messages-part-1-44ec179a5f45"

---

The NFC Data Exchange Format ([NDEF](https://developer.android.com/reference/android/nfc/NdefMessage)) is a standardized data format that can be used to exchange information between any compatible NFC device and another NFC device or tag. The data format consists of NDEF Messages and NDEF Records.

In this series of articles, I will explain how an NDEF message can be constructed and stored on an NFC tag. Assume a company wants to issue tags that can be used in public transport systems as a replacement for paper tickets. These tags can be tapped on an NFC-enabled Android device which will scan the tag and register the entry and exit time of the user. I won’t be going into how the application logic should be constructed but will be dealing with how the required data can be put on the tags.

#### Which NFC tag is recommended?

In my experience, I found Mifare Ultralight C to be suitable in most of the standard use-cases. Here’s a link to its [datasheet](https://www.nxp.com/docs/en/data-sheet/MF0ICU2.pdf) which you might use when you start digging deeper into its specifications. If you are just getting started then you can buy the [10 pack of Mifare Ultralight C](https://amzn.to/2HS7m8P) from Amazon or if you want to order in bulk then get the [100 pack](https://amzn.to/3kKCeGj) instead.

Apart from Ultralight C, I found Mifare Classic 1k to be a great choice for some scenarios when your data size is larger. Here’s a link to its [datasheet](https://www.nxp.com/docs/en/data-sheet/MF1S50YYX_V1.pdf) for more details. You can easily get a [pack of 100 cards](https://amzn.to/37W9pU0) from Amazon.

In this article let's get an overview of how NDEF messages are constructed.

### NDEF Message

NDEF messages are the basic transport mechanism for NDEF records, with an NDEF message containing one or more NDEF records.

For this use case the tag primarily needs the following records on it:

- A user identifier, say username to identify the user.
- An application identifier, to tell the android system which app will be handling the NFC tag.

### NDEF Record

NDEF Records contain a specific payload and have the following structure that identifies the contents and size of the record. The basic elements that construct an NDEF record are:

- TNF ie. Type Name Format Field
- `type` which is a value corresponding to bits set in the TNF field
- ID value
- payload value

More specifically an NDEF record can be composed in the following way.

```
Bit 7     6       5       4       3       2       1       0
------  ------  ------  ------  ------  ------  ------  ------ 
[ MB ]  [ ME ]  [ CF ]  [ SR ]  [ IL ]  [        TNF         ]

[                         TYPE LENGTH                        ]

[                       PAYLOAD LENGTH                       ]

[                          ID LENGTH                         ]

[                         RECORD TYPE                        ]

[                              ID                            ]

[                           PAYLOAD                          ]
```

#### NDEF record explained

An NDEF record starts with 8 bits header that describes the record.

**TNF:**The Type Name Format or TNF Field of an NDEF record is a 3-bit value that describes the record type.

```
|       Type        |                                    Description                                    | Type Value |
|-------------------|-----------------------------------------------------------------------------------|------------|
| TNF_EMPTY         | Indicates the record is empty.                                                    | 0x00       |
| TNF_WELL_KNOWN    | Indicates the type field contains a well-known RTD type name.                     | 0x01       |
| TNF_MIME_MEDIA    | Indicates the type field contains a media-type                                    | 0x02       |
| TNF_ABSOLUTE_URI  | Indicates the type field contains an absolute-URI                                 | 0x03       |
| TNF_EXTERNAL_TYPE | Indicates the type field contains an external type name                           | 0x04       |
| TNF_UNKNOWN       | Indicates the payload type is unknown                                             | 0x05       |
| TNF_UNCHANGED     | Indicates the payload is an intermediate or final chunk of a chunked NDEF Record. | 0x06       |
```

**IL [ID Length bit]:**The IL flag indicates if the ID Length Field is present or not.

**SR [Short Record bit]:**The SR flag is set to 1 if the payload length field is 1 byte (8 bits/0–255) or less. This allows for more compact records.

**CF [Chunk Flag bit]:**The CF flag indicates if this is the first record chunk or a middle record chunk. For the 1st record of the message, it is set to 0 and for subsequent records, it is set to 1.

**ME [Message End bit]:**The ME flag indicates if this is the last record in the message. It is set to 1 for the last record.

**MB [Message Begin bit]:**The MB flag indicates if this is the first record in the message. It is set to 1 for the first message.

#### NDEF Records

For our application, we need the following NDEF records.

**Tag data**

- **tnf**: TNF_MIME_MEDIA,
- **type**: application/vnd.com.tickets [in bytes]
- **id**: null
- **payload**: byjwdH_6DNsuU4iTSrVaNEe6e52VLhhr4v_iGTRP7jQ= [in bytes]

**Android Application Record(AAR)**

- **tnf**: TNF_EXTERNAL_TYPE
- **type**: android.com:pkg [in bytes]
- **id**: null
- **payload**: com.example.tickets [in bytes]

While working on Android, the following [constructor](https://developer.android.com/reference/android/nfc/NdefRecord#NdefRecord%28short,%20byte[],%20byte[],%20byte[]%29) can be used to construct an `NdefRecord`.

```
public NdefRecord (short tnf, 
                byte[] type, 
                byte[] id, 
                byte[] payload)
```

With multiple `NdefRecord` you can construct an `NdefMessage`.

```
public NdefMessage (NdefRecord record, 
                NdefRecord... records)
```

We have seen an overview of the different parts of an NDEF message. In the next part of this article, we will see how these two NDEF records can be written on a tag using Android’s NFC capabilities.

We will also discuss the memory structure of two kinds of tags ie. Mifare Ultralight C and Mifare Classic.

You can buy me a coffee if this post really helped you learn something or fix a nagging issue!

* * *
Written on May 20, 2018 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/understanding-the-format-of-ndef-messages-part-1-44ec179a5f45)
