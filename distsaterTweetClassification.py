# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

### Read Train Data
# path = '../input/nlp-getting-started/train.csv'
path = './nlp-getting-started/train.csv'
data = pd.read_csv(path)

'''
Submission 1: Use just bag of words approach from just the tweets to predict

* Use only unique words
* Use only alpha numeric characters
'''

data = data.loc[:,['text','target']]

## Tweet cleaning 
## Convert all to lower case chars
## Keep only alpha numerics

def keepAlphaNumeric(text: str):
    return re.sub("[^a-z0-9 ]+","",text.lower())

data.loc[:,'text'] = data.loc[:,'text'].apply(keepAlphaNumeric)