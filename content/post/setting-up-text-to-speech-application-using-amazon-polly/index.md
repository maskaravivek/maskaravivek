---
title: "Setting up Text To Speech Application using Amazon Polly"
author: "Vivek Maskara"
date: 2018-03-17T12:12:23.340Z
lastmod: 2021-10-29T21:23:32-07:00

description: ""

subtitle: ""

tags: [AWS, Text to speech, Amazon Polly]
categories: [AWS]

image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_0.png"
 - "/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_1.png"
 - "/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_2.png"
 - "/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_3.png"
 - "/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_4.png"
 - "/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_5.png"
 - "/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_6.png"
 - "/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_7.png"
 - "/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_8.png"
 - "/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_9.png"
 - "/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_10.png"
 - "/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_11.png"
 - "/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_12.png"
 - "/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_13.png"
 - "/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_14.png"
 - "/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_15.png"
 - "/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_16.png"
 - "/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_17.png"
 - "/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_18.png"
 - "/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_19.png"
 - "/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_20.png"
 - "/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_21.png"
 - "/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_22.png"


aliases:
- "/setting-up-text-to-speech-applicaion-using-amazon-polly-3b67d6eef59c"

---

Long back we used [AWS to set up a PHP and MYSQL application](/web-hosting-using-php-and-mysql-on-aws-95bd5df0bd75). This weekend will work with [Amazon Polly](https://aws.amazon.com/polly/) to deploy our own TTS application.

> Amazon Polly is a Text-to-Speech service that uses advanced deep learning technologies to synthesize speech that sounds like a human voice.

This article is heavily based on the [guide on AWS](https://aws.amazon.com/blogs/machine-learning/build-your-own-text-to-speech-applications-with-amazon-polly/). Our application architecture looks as below:

![](/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_0.png#layoutTextWidth)

Let us start building the application where we will set up a few lambda functions, SNS, and S3 bucket to finally result in RESTful APIs that can convert text to speech for us.

#### Create a DynamoDB Table

Create a **DynamoDB**table to store text and corresponding audio files.

![](/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_1.png#layoutTextWidth)

#### Create an S3 Bucket

Create an **S3**bucket that will hold all the audio files for you. Go through the **Create bucket**wizard to complete the process.

![](/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_2.png#layoutTextWidth)

#### Create a SNS Topic

The work of converting a text file to an audio output would be done by 2 Lambda functions. Let’s create a new SNS topic from the SNS console.

![](/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_3.png#layoutTextWidth)

#### Create a New Role

Create a new **Role**in the **IAM Console**.

![](/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_4.png#layoutTextWidth)

After choosing the service that will use this role, go ahead and give this **Role**a name and click on **Create Role**.

![](/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_5.png#layoutTextWidth)

After the **Role**is created, click on **Add inline policy**under the **Permissions**tab.

![](/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_6.png#layoutTextWidth)

Copy paste the following policy, which provides Lambda with access to the services included in the architecture diagram

```
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "polly:SynthesizeSpeech",
        "dynamodb:Query",
        "dynamodb:Scan",
        "dynamodb:PutItem",
        "dynamodb:UpdateItem",
        "sns:Publish",
        "s3:PutObject",
        "s3:PutObjectAcl",
        "s3:GetBucketLocation",
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": [
        "*"
      ]
    }
  ]
}
```

After adding the JSON, review the policy and name it.

![](/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_7.png#layoutTextWidth)

#### **Creating a New Post Lambda Function**

![](/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_8.png#layoutTextWidth)

Copy paste the following code for it:

```
import boto3
import os
import uuid

def lambda_handler(event, context):
    
    recordId = str(uuid.uuid4())
    voice = event["voice"]
    text = event["text"]

print('Generating new DynamoDB record, with ID: ' + recordId)
    print('Input Text: ' + text)
    print('Selected voice: ' + voice)
    
    #Creating new record in DynamoDB table
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(os.environ['DB_TABLE_NAME'])
    table.put_item(
        Item={
            'id' : recordId,
            'text' : text,
            'voice' : voice,
            'status' : 'PROCESSING'
        }
    )
    
    #Sending notification about new post to SNS
    client = boto3.client('sns')
    client.publish(
        TopicArn = os.environ['SNS_TOPIC'],
        Message = recordId
    )
    
    return recordId
```

Use the following environment variables for the DynamoDB table and the SNS topic.

- `SNS_TOPIC` — the Amazon Resource Name (ARN) of the SNS topic we created
- `DB_TABLE_NAME` — the name of the DynamoDB table (in our case, it’s posts)

Assign the IAM role that we created for the Lambda functions.

![](/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_9.png#layoutTextWidth)

Add a **New Test** in the wizard to test if the function is working.

```
{   "voice": "Joanna",   "text": "This is working!" }
```

Now you test your function by clicking on **Test**.

![](/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_10.png#layoutTextWidth)

#### Create a Convert to Audio Lambda function

Again use the same wizard to create a new Lambda function, `PostReader_ConvertToAudio`.

![](/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_11.png#layoutTextWidth)

Configure an SNS trigger so that this function is executed whenever a new post comes in.

![](/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_12.png#layoutTextWidth)

Copy-paste the following code to this lambda function editor. Take care of your indentations, its python. :D

```
import boto3
import os
from contextlib import closing
from boto3.dynamodb.conditions import Key, Attr

def lambda_handler(event, context):

postId = event["Records"][0]["Sns"]["Message"]
    
    print "Text to Speech function. Post ID in DynamoDB: " + postId
    
    #Retrieving information about the post from DynamoDB table
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(os.environ['DB_TABLE_NAME'])
    postItem = table.query(
        KeyConditionExpression=Key('id').eq(postId)
    )

text = postItem["Items"][0]["text"]
    voice = postItem["Items"][0]["voice"] 
    
    rest = text
    
    #Because single invocation of the polly synthesize_speech api can 
    # transform text with about 1,500 characters, we are dividing the 
    # post into blocks of approximately 1,000 characters.
    textBlocks = []
    while (len(rest) > 1100):
        begin = 0
        end = rest.find(".", 1000)

if (end == -1):
            end = rest.find(" ", 1000)
            
        textBlock = rest[begin:end]
        rest = rest[end:]
        textBlocks.append(textBlock)
    textBlocks.append(rest)

#For each block, invoke Polly API, which will transform text into audio
    polly = boto3.client('polly')
    for textBlock in textBlocks: 
        response = polly.synthesize_speech(
            OutputFormat='mp3',
            Text = textBlock,
            VoiceId = voice
        )
        
        #Save the audio stream returned by Amazon Polly on Lambda's temp 
        # directory. If there are multiple text blocks, the audio stream
        # will be combined into a single file.
        if "AudioStream" in response:
            with closing(response["AudioStream"]) as stream:
                output = os.path.join("/tmp/", postId)
                with open(output, "a") as file:
                    file.write(stream.read())

s3 = boto3.client('s3')
    s3.upload_file('/tmp/' + postId, 
      os.environ['BUCKET_NAME'], 
      postId + ".mp3")
    s3.put_object_acl(ACL='public-read', 
      Bucket=os.environ['BUCKET_NAME'], 
      Key= postId + ".mp3")

location = s3.get_bucket_location(Bucket=os.environ['BUCKET_NAME'])
    region = location['LocationConstraint']
    
    if region is None:
        url_begining = "https://s3.amazonaws.com/"
    else:
        url_begining = "https://s3-" + str(region) + ".amazonaws.com/" \
    
    url = url_begining \
            + str(os.environ['BUCKET_NAME']) \
            + "/" \
            + str(postId) \
            + ".mp3"

#Updating the item in DynamoDB
    response = table.update_item(
        Key={'id':postId},
          UpdateExpression=
            "SET #statusAtt = :statusValue, #urlAtt = :urlValue",                   
          ExpressionAttributeValues=
            {':statusValue': 'UPDATED', ':urlValue': url},
        ExpressionAttributeNames=
          {'#statusAtt': 'status', '#urlAtt': 'url'},
    )
        
    return
```

Use the following environment variables and values:

- `DB_TABLE_NAME` — The name of the DynamoDB table (in our case, it’s posts )
- `BUCKET_NAME` — The name of the S3 bucket that we created to store MP3 files

Update the execution time settings for this lambda function.

![](/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_13.png#layoutTextWidth)

Go ahead and retest the `PostReader_NewPost` function. It will add an entry to your database table and a `mp3` file to the S3 bucket.

#### Create a Lambda function to get Audio

Now, we just need to create another lambda function, `PostReader_GetPost` that retrieves information from the database.

![](/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_14.png#layoutTextWidth)

```
import boto3
import os
from boto3.dynamodb.conditions import Key, Attr

def lambda_handler(event, context):
    
    postId = event["postId"]
    
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(os.environ['DB_TABLE_NAME'])
    
    if postId=="*":
        items = table.scan()
    else:
        items = table.query(
            KeyConditionExpression=Key('id').eq(postId)
        )
    
    return items["Items"]
```

This function needs just one environment variable ie. `DB_TABLE_NAME`.

To test function add a **New Test**as below:

```
{   "postId": "*" }
```

#### Exposing the Lambda function as an API

Next, we need to expose our application logic as a RESTful web service. Choose to **Create API**option in the **API Gateway**console to get started.

![](/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_15.png#layoutTextWidth)

After the API is created, choose **Create Method**from the menu.

![](/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_16.png#layoutTextWidth)

The `POST` method invokes the `PostReader_NewPost` Lambda function. For the `GET` method, our API invokes the `PostReader_GetPost` Lambda function.

Next, enable `CORS` to enable calling these methods from a different domain.

![](/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_17.png#layoutTextWidth)

Now, modify the `GET` method to add a `postId` query param to it.

![](/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_18.png#layoutTextWidth)

The lambda function expects a JSON input so we need to add a mapping in the **Integration Request**section.

![](/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_19.png#layoutTextWidth)

Similarly, you can configure your `POST` method to accept a JSON input. Go to the **Models**section and create a new model.

![](/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_20.png#layoutTextWidth)

```
{
  "$schema" : "http://json-schema.org/draft-04/schema#",
  "title" : "AloudStory",
  "type" : "object",
  "properties" : {
    "voice" : { "type" : "string" },
    "text" : { "type" : "string" }
  }
}
```

#### Deploying the API

Now that you have configured the methods, go ahead and deploy it.

![](/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_21.png#layoutTextWidth)

Choose a deployment stage

![](/post/img/2018-03-17_setting-up-text-to-speech-application-using-amazon-polly_22.png#layoutTextWidth)

That’s it your APIs are up and running. The AWS console will show you the invocation URL for it.

* * *
Written on March 17, 2018 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/setting-up-text-to-speech-applicaion-using-amazon-polly-3b67d6eef59c)
