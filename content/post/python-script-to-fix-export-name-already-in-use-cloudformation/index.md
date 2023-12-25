---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Python Script to Fix \"Export EXPORT_NAME cannot be updated as it is in use by STACK_NAME\" error in AWS CloudFormation"
subtitle: ""
summary: ""
authors: [admin]
tags: [AWS, Python, Boto3, AWS CDK]
categories: [AWS, AWS CDK]
date: 2023-12-24T23:16:01-07:00
lastmod: 2023-12-24T23:16:01-07:00
featured: true
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: true

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

If you use [AWS CloudFormation](https://aws.amazon.com/cloudformation/), you might have encountered an issue while trying to update stacks if the stacks are updated. The error looks something like this:

```
Export EXPORT_NAME cannot be updated as it is in use by STACK_NAME
```

If you use AWS CDK, you might still face issues while deploying stacks, as AWS CDK generates a CloudFormation template during deployment. While working with AWS CDK, I have faced this issue multiple times, and until now, I have been following the [manual steps outlined in this post](https://repost.aws/knowledge-center/cloudformation-stack-export-name-error) to update the export values in the stack using the AWS Console. 

I recently wrote a Python script that uses [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) to automate this process, which has worked well for me. The script solves a very specific use case of replacing `Fn::ImportValue` usages with actual values. This takes care of replacing the `EXPORT_NAME` usage with the actual value. 

Let us go over the script and learn how to use it. First create a `main.py` file and add the following code snippet to it:

```python
import boto3
import sys
import json

export_name = sys.argv[1]

print(f'export_name: {export_name}')

client = boto3.client('cloudformation')

def get_export_value(export_name):
    paginator = client.get_paginator('list_exports')

    for page in paginator.paginate():
        for export in page['Exports']:
            if export['Name'] == export_name:
                export_value = export['Value']
                return export_value

    return None

def get_template(stack_name):
    response = client.get_template(StackName=stack_name)

    return response['TemplateBody']

def replace_fn_import_with_value(template, export_name, export_value):
    template_str = json.dumps(template)
    
    import_key = '{' + "\"Fn::ImportValue\": \"{name}\"".format(name = export_name) + '}'
    
    import_value = '"{export_value}"'.format(export_value = export_value)
    
    template_str = template_str.replace(import_key, import_value)
    
    template = json.loads(template_str)

    return template

def get_list_of_stacks_for_export_name(export_name):
    imports = client.list_imports(ExportName=export_name)

    return imports['Imports']

def update_stack(stack_name, template):
    response = client.update_stack(
        StackName=stack_name,
        Capabilities=[
            'CAPABILITY_IAM',
            'CAPABILITY_NAMED_IAM',
            'CAPABILITY_AUTO_EXPAND'
        ],
        TemplateBody=json.dumps(template)
    )

    print("Stack updated:", stack_name)
    return response

def fetch_and_update_stack(stack_name, export_name, export_value):
    template = get_template(stack_name)
    updated_template = replace_fn_import_with_value(template, export_name, export_value)
    update_stack(stack_name, updated_template)

def update_stacks():
    export_value = get_export_value(export_name)
    imported_stacks = get_list_of_stacks_for_export_name(export_name)
    
    print('Export value: ' + export_value)

    for stack in imported_stacks:
        print('Stack name: ' + stack)
        fetch_and_update_stack(stack, export_name, export_value)
    
update_stacks()
```

The script performs the following operations:
- First, it reads the `export_name` from command line arguments.
- Next, it fetches the actual export value for the `export_name` using the `get_export_value` function
- Next, it fetches a list of all stacks that use `export_name` using the `get_list_of_stacks_for_export_name` function.
- Finally, for every stack, the `fetch_and_update_stack` is invoked to replace `export_name` with the actual value. Notice that the function uses `replace_fn_import_with_value` function to manipulate the template JSON for replacing the value.

You can run the script using the following command:

```
python main.py <YOUR_EXPORT_NAME>
```

Executing the script will replace your export in all the stacks that use it. After executing the script, you can continue with your AWS CDK or CloudFormation deployment and it should proceed without any issues. 

Note: The script assumes that you have [configured AWS credentials](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html) on your shell before executing it. 

That's it for this post. I hope you find this post useful! 