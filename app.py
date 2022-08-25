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

import streamlit as st
from PIL import Image

from model_methods import predict
import base64 # for title image

# configuration of the page
st.set_page_config(
    layout='centered',
    page_icon=Image.open('subreddit_icon.png'),
    page_title='Marvel vs. DC comics',
    initial_sidebar_state='auto'
)

def main():
    # embed source link in title image using base64 module
    # reference: https://discuss.streamlit.io/t/how-to-show-local-gif-image/3408/4
    # reference: https://discuss.streamlit.io/t/local-image-button/5409/4
    im = open("subreddit_icon.png", "rb")
    contents = im.read()
    im_base64 = base64.b64encode(contents).decode("utf-8")
    im.close()
    html = f'''<a href='https://www.reddit.com/'> 
            <img src='data:image/png;base64,{im_base64}' width='100'>
            </a><figcaption>Credit: reddit.com</figcaption>'''
    st.markdown(html, unsafe_allow_html=True)

    st.title('Subreddit Post classifier')
    local_css("highlight_text.css")
    text = '''The algorithm driving this app is built using subreddit posts published 
    between April and July 2022. It is only able to classify between 
    <span class='highlight blue'> **Marvel** </span>
    and 
    <span class='highlight blue'> **DC Comics** </span>
    subreddits.'''
    st.markdown(text, unsafe_allow_html=True)

    # Area for text input
    st.markdown('''
    Please copy and paste the 
    subreddit post here
    ''')

    import_nltk() # import nltk module if not yet cached in local computer
    new_post = st.text_input('Enter text here', '')

    data = pd.Series(new_post) # pd.Series format new input coz that is the format that predict() recognises
    
    # instantiate submit button
    if st.button('Submit'):
        with st.sidebar:
            try: 
                # process new input
                result = predict(data)
                if result == 1:
                    post = 'Marvel'
                if result == 0:
                    post = 'DC comics'
                st.write(f'### This post belongs to') 
                st.success(f'# {post}')
                st.write(f'### subreddit')
            except:
                st.warning('''
                Unable to detect text. 
                Please enter text for prediction. 
                \n\n Thank you 🙏. 
                ''')

#######################################################################
@st.cache # cached so that latency for subsequent runs are shorter
def import_nltk():
    import nltk
    nltk.download('wordnet')
    nltk.download('omw-1.4')
# https://discuss.streamlit.io/t/are-you-using-html-in-markdown-tell-us-why/96/25
def local_css(file_name): # for highlighting text
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
def ML_model():
    # import dataset 'full_post' that has been lemmatized
    df = pd.read_csv('tts.csv')
    # train-test-split
    X = df['full_post'] # pd.series because dataframe format not friendly for word vectorization
    y = df['subreddit']
    # make sure target variable has equal representation on both train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=.2,
                                                        stratify=y, 
                                                        random_state=42)
    Z_train = X_train.apply(lemmatize_join)
    # model instantiation
    pipe_cvec_nb = Pipeline([
                            ('cvec', CountVectorizer()),
                            ('nb', MultinomialNB())
                            ])
    # word vectorizor hyperparameter tuning
    features = [1000]
    min_df = [3]
    max_df = [.6]
    ngrams = [(1,2)]
    stop_words = ['english']
    accent = ['unicode']
    # naive bayes classifier parameters
    alphas = [.5]
    # input parameters
    cvec_nb_params = [{'cvec__max_features': features,
                       'cvec__min_df': min_df,
                       'cvec__max_df': max_df,
                       'cvec__ngram_range': ngrams,
                       'cvec__lowercase': [False],
                       'cvec__stop_words': stop_words,
                       'cvec__strip_accents': accent,
                       'nb__alpha': alphas
                      }]
    # instantiate GridSearch
    cvec_nb_gs = GridSearchCV(pipe_cvec_nb,
                              cvec_nb_params,
                              scoring='accuracy',
                              cv=5,
                              verbose=1,
                              n_jobs=-2)
    # fit model
    model = cvec_nb_gs.fit(Z_train, y_train)
    return model
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
    df = pd.read_csv('tts.csv')

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
    # load model
    pred = ML_model().predict(Z_data)
    return pred
if __name__=='__main__':
    main()