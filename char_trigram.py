#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


ar_corpus = 'arabizi-english-bitext.arz'
leb_corpus = 'arabizi-twitter-leb.csv'
egy_corpus = 'arabizi-twitter-egy.csv'
en_twitter_corpus = 'arab_politics_tweet.txt'
### Change en_corpus to a true English (non-translated) social media dataset
en_corpus = 'arabizi-english-bitext.en'


def make_sentence_df(corpus, sep, header, target=None):
    """
    """
    if target != None:
        sentence_df = pd.read_csv(corpus, sep=sep, header=header, \
                              names=['sentence'])
        sentence_df['language'] = target
    else:
        sentence_df = pd.read_csv(corpus, sep=sep, header=header, \
                              names=['sentence', 'language'])

    if corpus == 'arabizi-twitter-egy.csv':
        sentence_df.loc[4064, 'language'] = 0
        sentence_df['language'] = sentence_df['language'].astype(int)
    print(corpus, 'dtype:', sentence_df.language.dtype)
    
    return sentence_df


def combine_sentence_df(sentence_df1, sentence_df2):
    """
    """
    sentence_df = sentence_df1.append(sentence_df2, ignore_index=True)
    
    return sentence_df


def char_vectorize(sentence_df, n=3):
    """
    """
    y = sentence_df.language
    X_train, X_test, y_train, y_test = \
    train_test_split(sentence_df['sentence'], y, test_size=0.1, random_state=0) 
    
    count_vectorizer = CountVectorizer(analyzer = 'char', ngram_range = (3, 3))
    count_train = count_vectorizer.fit_transform(X_train.values)
    count_test = count_vectorizer.transform(X_test.values)
    
    return count_train, count_test, y_train, y_test
 

def naive_bayes(count_train, count_test, y_train, y_test):
    """
    """
    nb_classifier = MultinomialNB()
    nb_classifier.fit(count_train, y_train)
    pred = nb_classifier.predict(count_test)
    
    score = metrics.accuracy_score(y_test, pred)
    print(score)
    
    confusion_mtrx = metrics.confusion_matrix(y_test, pred, labels=[1, 0])
    print(confusion_mtrx)

    return score, confusion_mtrx


def support_vector_machine(count_train, count_test, y_train, y_test):
    """
    """
    svm_classifier = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    svm_classifier.fit(count_train, y_train)
    pred = svm_classifier.predict(count_test)
    
    score = metrics.accuracy_score(y_test, pred)
    print(score)
    
    confusion_mtrx = metrics.confusion_matrix(y_test, pred, labels=[1, 0])
    print(confusion_mtrx)
    
    return score, confusion_mtrx


ar_text = make_sentence_df(ar_corpus, '\n', header=None, target=1)
en_text = make_sentence_df(en_corpus, '\n', header=None, target=0)
leb_text = make_sentence_df(leb_corpus, ',', header=0)
egy_text = make_sentence_df(egy_corpus, ',', header=0)
en_twitter_text = make_sentence_df(en_twitter_corpus, '\n', header=0, target=0)

textc = combine_sentence_df(ar_text, en_text)
texta = combine_sentence_df(en_twitter_text, textc)
textb = combine_sentence_df(leb_text, egy_text)
text = combine_sentence_df(texta, textb)

X_train, X_test, y_train, y_test = char_vectorize(text)
nb = naive_bayes(X_train, X_test, y_train, y_test)
svm = support_vector_machine(X_train, X_test, y_train, y_test)