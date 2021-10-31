---
title: "Working with NFC tags on Android"
author: "Vivek Maskara"
date: 2020-03-16T18:52:36.608Z
lastmod: 2021-10-29T21:24:58-07:00

description: ""

subtitle: ""

categories: [Android]

tags:
 - Android
 - Near Field Communication
 - Contactless Payments
 - Android App Development
 - Mifare


image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2020-03-16_working-with-nfc-tags-on-android_0.jpeg"
 - "/post/img/2020-03-16_working-with-nfc-tags-on-android_1.png"


aliases:
- "/working-with-nfc-tags-on-android-c1e5af47a3db"

---

![](/post/img/2020-03-16_working-with-nfc-tags-on-android_0.jpeg#layoutTextWidth)

In this post, I will show you how to read and write an NFC tag on an Android device. We would be using [Android’s NFC capabilities](https://developer.android.com/guide/topics/connectivity/nfc/nfc) to read and write a tag. In a different post, I will illustrate how APDU commands could be used to talk directly with an NFC tag.

### Prerequisites

You will need an NFC capable Android device and NFC tags before you get started with this tutorial.

In my experience, I found Mifare Ultralight C to be suitable in most of the standard use-cases. Here’s a link to its [datasheet](https://www.nxp.com/docs/en/data-sheet/MF0ICU2.pdf) which you might use when you start digging deeper into its specifications. If you are just getting started then you can buy the [10 pack of Mifare Ultralight C](https://amzn.to/2HS7m8P) from Amazon or if you want to order in bulk then get the [100 pack](https://amzn.to/3kKCeGj) instead.

Apart from Ultralight C, I found Mifare Classic 1k to be a great choice for some scenarios when your data size is larger. Here’s a link to its [datasheet](https://www.nxp.com/docs/en/data-sheet/MF1S50YYX_V1.pdf) for more details. You can easily get a [pack of 100 cards](https://amzn.to/37W9pU0) from Amazon.

### Introduction

Earlier I talked about the structure of NDEF messages. You can go through this post to understand the format of NDEF messages.

[Understanding the format of NDEF Messages — Part 1](/understanding-the-format-of-ndef-messages-part-1-44ec179a5f45 "https://medium.com/@maskaravivek/understanding-the-format-of-ndef-messages-part-1-44ec179a5f45")

Broadly these actions are performed when a tag is tapped on an Android device:

- NFC tag is handled with the tag dispatch system, which analyses discovered NFC tags
- it categorizes the data and starts an application that is interested in the categorized data
- an application that wants to handle the scanned NFC tag can declare an intent filter and request to handle the data

### Request NFC access for your app

Firstly, to access NFC hardware you need to add the following permission in `AndroidManifest.xml`

```
<uses-permission android:name="android.permission.NFC" />
```

Note: If you want your application to show up **only** for devices with NFC hardware then add the following `uses-feature` element in the manifest file.

```
<uses-feature android:name="android.hardware.nfc" android:required="true" />
```

### Setup Foreground Dispatch

You need to register foreground dispatch so that your activity can handle the NFC intents.

**Initialize NFC Adapter**

First, init the `NFCAdapter` in the `onCreate` method of the `Activity` or a similar `Fragment` lifecycle method.

```
private var adapter: NfcAdapter? = null

override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_main)
    .
    .
    .
    initNfcAdapter()
}

private fun initNfcAdapter() {
    val nfcManager = getSystemService(Context.NFC_SERVICE) as NfcManager
    adapter = nfcManager.defaultAdapter
}

```

**Enable Foreground Dispatch**

Next, call `enableForegroundDispatch` to enable foreground dispatch of the `NFCAdapter`. You can call this method in the `onResume` lifecycle method of the activity.

```
override fun onResume() {
    super.onResume()
    enableNfcForegroundDispatch()
}

private fun enableNfcForegroundDispatch() {
    try {
        val intent = Intent(this, javaClass).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP)
        val nfcPendingIntent = PendingIntent.getActivity(this, 0, intent, 0)
        adapter?.enableForegroundDispatch(this, nfcPendingIntent, null, null)
    } catch (ex: IllegalStateException) {
        Log.e(getTag(), "Error enabling NFC foreground dispatch", ex)
    }
}
```

**Disable Foreground Dispatch**

When the activity is paused or destroyed, make sure that you disable the foreground dispatch.

```
override fun onPause() {
    disableNfcForegroundDispatch()
    super.onPause()
}

private fun disableNfcForegroundDispatch() {
    try {
        adapter?.disableForegroundDispatch(this)
    } catch (ex: IllegalStateException) {
        Log.e(getTag(), "Error disabling NFC foreground dispatch", ex)
    }
}
```

### Prepare your NDEF records

We will take the example from the previous post and try to write the same data on an NFC tag.

**Custom data**

- **tnf**: TNF_MIME_MEDIA,
- **type**: application/vnd.com.tickets [in bytes]
- **id**: null
- **payload**: cxwwhcfxympwbbonxymwritcqytcnfvgmwcnzfanqytc [in bytes]

```
val typeBytes = mimeType.toByteArray()
val payload = tagData.toByteArray()
val r1 = NdefRecord(TNF_MIME_MEDIA, typeBytes, null, payload)
```

**Android Application Record(AAR)**

- **tnf**: TNF_EXTERNAL_TYPE
- **type**: android.com:pkg [in bytes]
- **id**: null
- **payload**: com.example.tickets [in bytes]

```
val r2 = NdefRecord.createApplicationRecord(context.packageName)
```

Construct your `NdefMessage` using both the records.

```
NdefMessage(arrayOf(record1, record2))
```

### Write NDEF message to the NFC tag

Now, that we have the foreground dispatch setup and the NDEF message prepared, we are ready to write the message on the NFC tag.

#### Listen to NFC intent

Listen to NFC intent and when a `EXTRA_TAG` data is present in the incoming intent, handle it to write to an NFC tag.

```
val tagFromIntent = intent.getParcelableExtra<Tag>(NfcAdapter.EXTRA_TAG)
try {
    tag = WritableTag(tagFromIntent)
} catch (e: FormatException) {
    Log.e(getTag(), "Unsupported tag tapped", e)
    return
}
```

#### Write the NDEF messages

The final step is to actually write the information on the tag. I have extracted out all the code to a helper class called `WritableTag` to abstract out all the logic of writing on an NFC tag to a separate class.

```
class WritableTag @Throws(FormatException::class) constructor(tag: Tag) {
    private val NDEF = Ndef::class.java.canonicalName
    private val NDEF_FORMATABLE = NdefFormatable::class.java.canonicalName

    private val ndef: Ndef?
    private val ndefFormatable: NdefFormatable?

    val tagId: String?
        get() {
            if (ndef != null) {
                return bytesToHexString(ndef.tag.id)
            } else if (ndefFormatable != null) {
                return bytesToHexString(ndefFormatable.tag.id)
            }
            return null
        }

    init {
        val technologies = tag.techList
        val tagTechs = Arrays.asList(*technologies)
        if (tagTechs.contains(NDEF)) {
            Log.i("WritableTag", "contains ndef")
            ndef = Ndef.get(tag)
            ndefFormatable = null
        } else if (tagTechs.contains(NDEF_FORMATABLE)) {
            Log.i("WritableTag", "contains ndef_formatable")
            ndefFormatable = NdefFormatable.get(tag)
            ndef = null
        } else {
            throw FormatException("Tag doesn't support ndef")
        }
    }

    @Throws(IOException::class, FormatException::class)
    fun writeData(tagId: String,
                  message: NdefMessage): Boolean {
        if (tagId != tagId) {
            return false
        }
        if (ndef != null) {
            ndef.connect()
            if (ndef.isConnected) {
                ndef.writeNdefMessage(message)
                return true
            }
        } else if (ndefFormatable != null) {
            ndefFormatable.connect()
            if (ndefFormatable.isConnected) {
                ndefFormatable.format(message)
                return true
            }
        }
        return false
    }

    @Throws(IOException::class)
    private fun close() {
        ndef?.close() ?: ndefFormatable?.close()
    }

    companion object {
        fun bytesToHexString(src: ByteArray): String? {
            if (ByteUtils.isNullOrEmpty(src)) {
                return null
            }
            val sb = StringBuilder()
            for (b in src) {
                sb.append(String.format("%02X", b))
            }
            return sb.toString()
        }
    }
}
```

Note:

- When the tag is read, we iterate through all the supported technologies of the tag to check if it supports`NDEF` or `NDEF_FORMATABLE` . If not, we throw a `FormatException` as we can’t write a NDEF message on an unsupported tag.
- You can check if your tag supports NFC or not by using any of the NFC reader apps from the play store. I prefer [this](https://play.google.com/store/apps/details?id=com.wakdev.wdnfc) app.
- If the tag is compatible, we simple `connect` to the tag and call `writeNdefMessage` to write the message on the tag.

### Reading NDEF message from an NFC Tag

Now that we have written our NDEF message on the NFC tag, we would probably want to read it. Also, it might be useful to read the tag UID for uniquely identifying the tag. Let us see how to do that.

#### Handle NFC Intent

To read the NFC tag, the app needs to register for handling `ACTION_NDEF_DISCOVERED` intent. Registering this intent will let your app handle any NFC tag that is tapped to the Android device.

```
  if (NfcAdapter.ACTION_NDEF_DISCOVERED == intent.action) {
      val rawMsgs = intent.getParcelableArrayExtra(NfcAdapter.EXTRA_NDEF_MESSAGES)
      if (rawMsgs != null) {
          onTagTapped(NfcUtils.getUID(intent), NfcUtils.getData(rawMsgs))
      }
  }
```

If you wish to handle only those tags that belong to your application then you can add a filter. Earlier we created an application record while writing the tag. We can use the same package name as the filter if the app needs to handle only those tags that have that particular application record(AAR).

```
fun getIntentFilters(): Array<IntentFilter> {
    val ndefFilter = IntentFilter(NfcAdapter.ACTION_NDEF_DISCOVERED)
    try {
        ndefFilter.addDataType("application/vnd.com.tickets")
    } catch (e: IntentFilter.MalformedMimeTypeException) {
        Log.e("NfcUtils", "Problem in parsing mime type for nfc reading", e)
    }

    return arrayOf(ndefFilter)
}

```

#### Reading Tag UID

You can read the tag ID of the tag using the following method.

```
fun getUID(intent: Intent): String {
    val myTag = intent.getParcelableExtra<Tag>(NfcAdapter.EXTRA_TAG)
    return BaseEncoding.base16().encode(myTag.id)
}
```

#### Read Tag Data

You can read all the NDEF records using the following code snippet.

```
fun getData(rawMsgs: Array<Parcelable>): String {
  val msgs = arrayOfNulls<NdefMessage>(rawMsgs.size)
  for (i in rawMsgs.indices) {
      msgs[i] = rawMsgs[i] as NdefMessage
  }

  val records = msgs[0]!!.records

  var recordData = ""

  for (record in records) {
      recordData += record.toString() + "\n"
  }

  return recordData
}
```

### Conclusion

Android makes it quite easy to read and write NFC tags and it supports a variety of tags and tag technologies. Once you understand the basics, you can build your own NFC supported app within a few hours.

You can find the complete source code of the example that I have used in this article on Github.

[maskaravivek/AndroidNfcExample](https://github.com/maskaravivek/AndroidNfcExample "https://github.com/maskaravivek/AndroidNfcExample")

You can buy me a coffee if this post really helped you learn something or fix a nagging issue!

* * *
Written on March 16, 2020 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/working-with-nfc-tags-on-android-c1e5af47a3db)
