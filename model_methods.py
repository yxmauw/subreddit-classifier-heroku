import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import re
import pickle

# lemmatizing
def lemmatize_join(text):
    tokenizer = RegexpTokenizer('[a-z]+', gaps=False) # instantiate tokenizer
    lemmer = WordNetLemmatizer() # instantiate lemmatizer
    return ' '.join([lemmer.lemmatize(w) for w in tokenizer.tokenize(text.lower())]) 
    # lowercase, join back together with spaces so that word vectorizers can still operate 
    # on cell contents as strings

def predict(new_data):
    # lemmatize new data
    Z_data = new_data.apply(lemmatize_join)

    # countvectorize new data
    # import dataset 'full_post' that has been lemmatized
    url = 'https://raw.githubusercontent.com/yxmauw/General_Assembly_Pub/main/project_3/cloud_app/tts.csv'
    df = pd.read_csv(url, header=0)

    # train-test-split
    X = df['full_post'] # pd.series because dataframe format not friendly for word vectorization
    y = df['subreddit']

    # make sure target variable has equal representation on both train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=.2,
                                                    stratify=y, 
                                                    random_state=42)
    cvec = CountVectorizer()
    Z_train = X_train.apply(lemmatize_join) # lemmatize training data
    cvec.fit(Z_train) # fit on lemmatized training data set
    cvec.transform(Z_data) # transform new data

    with open('project_3/cloud_app/final_model.sav','rb') as f:
        model = pickle.load(f)
    pred = model.predict(Z_data)
    return pred
