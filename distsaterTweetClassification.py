# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

def pprint(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None,'expand_frame_repr', True , 'display.max_colwidth', -1, 'display.width', 1000):
        print(df)

### Read Train Data
# path = '../input/nlp-getting-started/train.csv'
import os
os.chdir('/mnt/d/Projects/Disaster Tweet Classification/')

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

# def keepAlphaNumeric(text: str):
#     return re.sub("[^a-z0-9 ]+","",text.lower())

# def removeHyperLinks(text: str):
# 	return re.sub(" +", " ", re.sub("https?://\S+","",text))

# def removeMentions(text: str):
# 	return re.sub(" +", " ", re.sub("[\r\n]|@\\S+","", text))

from nltk.corpus import stopwords

def preprocess(text: str):
	text = " ".join([x for x in text.split(" ") if x not in stopwords.words("english")])
	text = re.sub(" +", " ", re.sub("https?://\S+","",text))
	text = re.sub(" +", " ", re.sub("[\r\n]|@\\S+","", text))
	text = re.sub(" +"," ",re.sub("[^a-z0-9 ]+","",text.lower()))
	text = re.sub("^ ", "", text)
	return text



# Remove hyper links

data.loc[:,'text'] = data.loc[:,'text'].apply(preprocess)


from sklearn.model_selection import train_test_split

train, val = train_test_split(data, test_size=0.2, random_state = 501)

train.reset_index(drop=True,inplace=True)
val.reset_index(drop=True,inplace=True)

# corpus = []
# for sentence in train['text']:
# 	corpus.extend(sentence.split(' '))
# 	corpus = list(set(corpus))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2)).fit(train['text'])

trainX = normalize(vectorizer.transform(train['text']), norm='l2',axis=1)

valX = normalize(vectorizer.transform(val['text']), norm='l2', axis=1)


# Model 1

import keras

# Structure: 6090:1024, 1024:512, 512:128, 128:64, 64:1

nnModel = keras.Sequential()
nnModel.add(keras.layers.Dense(1024, input_shape=(trainX.shape[1],), activation='relu'))
nnModel.add(keras.layers.Dropout(rate=0.6))
nnModel.add(keras.layers.Dense(1, activation='sigmoid'))
nnModel.compile(optimizer='adam', loss='binary_crossentropy')

es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
nnModel.fit(trainX, train['target'].values, verbose=1, epochs=10, batch_size=32, validation_data=(valX, val['target'].values),callbacks=[es_callback])

valPred = pd.concat([pd.DataFrame(nnModel.predict(valX), columns =['prediction']).loc[:,'prediction'].apply(lambda x: 1 if x> 0.5 else 0),val.loc[:,'target']],axis=1)

from sklearn.metric import f1_score

print("f1 score = " + str(f1_score(valPred['prediction'], valPred['target'])))
print("accuracy = "+ str(valPred.loc[valPred['prediction'] == valPred['target'],:].shape[0]/valPred.shape[0]))


def predictTest(model):
	test_path = './nlp-getting-started/test.csv'
	save_pred = './nlp-getting-started/pred.csv'
	test = pd.read_csv(test_path)
	test = test.loc[:,['id','text']]
	test.loc[:,'text'] = test.loc[:,'text'].apply(preprocess)
	testId = test.loc[:,['id']]
	testX = normalize(vectorizer.transform(test['text']), norm='l2',axis=1)
	submission = pd.concat([testId,pd.DataFrame(nnModel.predict(testX), columns=['target']).loc[:,'target'].apply(lambda x: 1 if x>0.5 else 0)],axis=1)
	submission.to_csv(save_pred, index=False)