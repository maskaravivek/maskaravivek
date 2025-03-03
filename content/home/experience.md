+++
# Experience widget.
widget = "experience"  # See https://sourcethemes.com/academic/docs/page-builder/
headless = true  # This file represents a page section.
active = true  # Activate this widget? true/false
weight = 40  # Order that this section will appear.

title = "Experience"
subtitle = ""

# Date format for experience
#   Refer to https://sourcethemes.com/academic/docs/customization/#date-format
date_format = "Jan 2006"

# Experiences.
#   Add/remove as many `[[experience]]` blocks below as you like.
#   Required fields are `title`, `company`, and `date_start`.
#   Leave `date_end` empty if it's your current employer.
#   Begin/end multi-line descriptions with 3 quotes `"""`.

[[experience]]
  title = "Associate Software Engineer"
  company = "JP Morgan"
  company_url = "https://www.linkedin.com/company/amazon/"
  location = "Jersey City, NJ"
  date_start = "2022-02-15"
  date_end = "2022-11-04"
  description = """
  Responsibilities include:
  
  Working as part of the Onyx team @ JPMC on machine learning and software development-related work.

  * Productionalized an internal intelligence platform by writing helm-charts for a stack containing Python apps, Dask, and PostgreSQL to a Kubernetes cluster.
  """

[[experience]]
  title = "Software Development Engineer Intern"
  company = "Amazon"
  company_url = "https://www.linkedin.com/company/amazon/"
  location = "Tempe, Arizona"
  date_start = "2020-05-17"
  date_end = "2020-08-06"
  description = """
  Responsibilities include:
  
  * Designed and developed a serverless system for verification of brand ID mappings capable of processing millions of records at a time. It utilizes multiple AWS services such as Lambda, SNS, SQS, Step functions, Glue Jobs, EventBridge, S3, and DynamoDB.
  * Programmatically provisioned the infrastructure using AWS CDK enabling CI/CD integration with the deployment pipeline. 
  * Optimized the AWS Glue Job to process ~40 million input records in <10 minutes performing validation against >1 billion ML output data.
  """

[[experience]]
  title = "Graduate Research Assistant"
  company = "The Luminosity Lab, ASU"
  company_url = "https://theluminositylab.com/"
  location = "Tempe, Arizona"
  date_start = "2020-03-02"
  date_end = "2021-12-13"
  description = """
  Responsibilities include:
  
  * Developing a machine learning model for detection of Neuroblastoma using histopathological images for PCH hospital. 
  * Researched and built the MVC for indoor parking automation using YOLO and DeepSort for real time vehicle tracking. 
  * Contributed to the backend for ASU’s end to end PPE response network for producing and delivering medical supplies.
  * Built a Customer 360 dashboard for Bank of West using Neo4J graph database, Flask backend and React for frontend. 
  * Published a gamified supply chain management learning app funded by USAID, ShipShape for iOS and Android. 
  """

[[experience]]
  title = "Senior Software Engineer"
  company = "Zeta, Directi"
  company_url = "https://zeta.tech"
  location = "Bangalore, India"
  date_start = "2016-06-06"
  date_end = "2019-11-19"
  description = """
  * Built NFC based contactless payments & custom ordering solution for POS attributing to 1 million+ monthly transactions.
  * Contributed to over 20+ projects in Zeta spanning across Android, Raspberry Pi and backend microservices. 
  * Responsible for optimizing query performance for PostgreSQL and building throughput and service health monitoring dashboards using Grafana and Kibana.
  """

+++
