import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import re

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

import pickle

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

# lemmatizing
def lemmatize_join(text):
    tokenizer = RegexpTokenizer('[a-z]+', gaps=False) # instantiate tokenizer
    lemmer = WordNetLemmatizer() # instantiate lemmatizer
    return ' '.join([lemmer.lemmatize(w) for w in tokenizer.tokenize(text.lower())]) 
    # lowercase, join back together with spaces so that word vectorizers can still operate 
    # on cell contents as strings

Z_train = X_train.apply(lemmatize_join)

# model instantiation
pipe_cvec_nb = Pipeline([
    ('cvec', CountVectorizer()),
    ('nb', MultinomialNB())
])

# word vectorizor parameters
features = [1000]
min_df = [3]
max_df = [.6]
ngrams = [(1,2)]
stop_words = ['english']
accent = ['unicode']

# naive bayes classifier parameters
alphas = [.5]

cvec_nb_params = [{'cvec__max_features': features,
                   'cvec__min_df': min_df,
                   'cvec__max_df': max_df,
                   'cvec__ngram_range': ngrams,
                   'cvec__lowercase': [False],
                   'cvec__stop_words': stop_words,
                   'cvec__strip_accents': accent,
                   'nb__alpha': alphas
                   }]

cvec_nb_gs = GridSearchCV(pipe_cvec_nb,
                          cvec_nb_params,
                          scoring='accuracy',
                          cv=5,
                          verbose=1,
                          n_jobs=-2)

cvec_nb_gs.fit(Z_train, y_train)

pickle.dump(cvec_nb_gs, open('final_model.sav', 'wb'))
