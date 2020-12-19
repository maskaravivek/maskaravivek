---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Tabular Synthetic Data Generation using CTGAN"
subtitle: ""
summary: ""
authors: [admin]
tags: [GANs, Privacy, Deep Learning, Synthetic Data]
categories: [Deep Learning, GANs]
date: 2020-12-18T20:00:21-07:00
lastmod: 2020-12-18T20:00:21-07:00
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

In this post we we will talk about generating synthetic data from tabular data using Generative adversarial networks(GANs). We will be using the default implementation of [CTGAN](https://github.com/sdv-dev/CTGAN) [1] model. 

![png](featured.png)

## Introduction

In the last post on GANs we saw how to generate synthetic data on Synthea dataset. Here's a link to the post for a refresher: 

https://www.maskaravivek.com/post/gan-synthetic-data-generation/

Similar to the last post, we would be working with the Synthea dataset which is publicly available. 

https://synthetichealth.github.io/synthea/

In this post, we will be working on the `patients.csv` file and will only be using continious and categorical fields. We will remove the other fields like name, email ID etc which contains a lot of unique values and will thus will be difficult to learn. 

## Data Preprocessing

Firstly, download the publicly available synthea dataset and unzip it. 


```
!wget https://storage.googleapis.com/synthea-public/synthea_sample_data_csv_apr2020.zip
!unzip synthea_sample_data_csv_apr2020.zip
```

## Install Dependencies

In this post, we will be using the default implementation of CTGAN which is available here. 

https://github.com/sdv-dev/CTGAN

To use CTGAN do a pip install. Also, we will be installing the `table_evaluator` library([link](https://pypi.org/project/table-evaluator/)) which will help us in comparing the results with the original data. 


```
!pip install ctgan
!pip install table_evaluator
```

### Remove unnecessary columns and encode all data

Next, we read the data into a dataframe and drop the unnecessary columns. 


```
import pandas as pd

data = pd.read_csv('csv/patients.csv')
data.drop(['Id', 'BIRTHDATE', 'DEATHDATE', 'SSN', 'DRIVERS', 'PASSPORT', 'PREFIX',
       'FIRST', 'ADDRESS', 'LAST', 'SUFFIX', 'MAIDEN','LAT', 'LON',], axis=1, inplace=True)
print(data.columns)
```

    Index(['MARITAL', 'RACE', 'ETHNICITY', 'GENDER', 'BIRTHPLACE', 'CITY', 'STATE',
           'COUNTY', 'ZIP', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE'],
          dtype='object')


Next, we define a list with column names for categorical variables. This list will be passed to the model so that the model can decide how to process these fields. 


```
categorical_features = ['MARITAL', 'RACE', 'ETHNICITY', 'GENDER', 'BIRTHPLACE', 'CITY', 'STATE', 'COUNTY', 'ZIP']
```

## Training the model

Next, we simply define an instance of `CTGANSynthesizer` and call the `fit` method with the dataframe and the list of categorical variables. 

We train the model for 300 epochs only as the discriminator and generator loss becomes quite low after these many epochs. 


```
from ctgan import CTGANSynthesizer

ctgan = CTGANSynthesizer(verbose=True)
ctgan.fit(data, categorical_features, epochs = 300)
```

## Evaluation

Next, we simply call model's `sample` function to generate samples based on the learned model. In this example we generate 1000 samples. 


```
samples = ctgan.sample(1000)

print(samples.head())
```

      MARITAL    RACE  ... HEALTHCARE_EXPENSES HEALTHCARE_COVERAGE
    0       S   asian  ...        7.331230e+05         8940.917593
    1     NaN   white  ...        1.540945e+06         3099.605568
    2     NaN   asian  ...        1.517647e+06        11947.241606
    3     NaN   white  ...        1.516137e+06        14091.349082
    4       S  native  ...        1.534122e+06         5103.408672
    
    [5 rows x 11 columns]


Now let's try to do a feature by feature comparision between the generated data and the actual data. We will use python's `table_evaluator` library to compare the features. 

We call the `visual_evaluation` method to compare the actual data(`data`) and the generated data(`samples`).


```
from table_evaluator import load_data, TableEvaluator

print(data.shape, samples.shape)
table_evaluator =  TableEvaluator(data, samples, cat_cols= categorical_features)

table_evaluator.visual_evaluation()
```

    (1171, 11) (1000, 11)


    
![png](images/CTGAN_Synthetic_data_generation_13_1.png)
    
![png](images/CTGAN_Synthetic_data_generation_13_2.png)
    
![png](images/CTGAN_Synthetic_data_generation_13_4.png)
  
![png](images/CTGAN_Synthetic_data_generation_13_5.png)
    
![png](images/CTGAN_Synthetic_data_generation_13_6.png)
    
## Conclusion

As its apparent from the visualizations, the similarity between the original data and the synthetic data is quite high. The results give a lot of confidence as we took a random dataset and applied the default implementation without any tweaks or any data preprocessing. 

The model can be used in various scenarios where data augmentation is required. Its worthwhile to highlight a few caveats:
- In this dataset we just had categorical and continuous variables and the results were quite good. 
- It would be useful to try it on datasets with date time values
- Also this model won't be able to handle relational datasets by default. For eg. there's no way of specifiying primary key foreign key constraints. 
- Moreover, it cannot handle contraints by default. For eg. a particular state should belong to a single country but there's no way of specifying this constraint. The generated dataset can contain new combinations of (state, country) which is not present in the original dataset. 

There's a framework to mitigate some of the above issues. Checkout [SDV](https://sdv.dev/SDV/) if you are interested. I will try to write a post about it in future.

## TL;DR

Here's the link to the Google colab notebook with the complete source code. 

https://colab.research.google.com/drive/1nwbvkg32sOUC69zATCfXOygFUBeo0dsx?usp=sharing


## References

[1] Lei Xu, Maria Skoularidou, Alfredo Cuesta-Infante, Kalyan Veeramachaneni. Modeling Tabular data using Conditional GAN. NeurIPS, 2019

