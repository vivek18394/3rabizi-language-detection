#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import re
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import sys
import time


ar_corpus = 'corpora/arabizi-english-bitext.arz'
ar_egy_corpus = 'corpora/arabizi-twitter-egy.csv'
#ar_leb_corpus = 'corpora/arabizi-leb-small.csv'
ar_leb_corpus = 'corpora/arabizi-twitter-leb.csv'
en_corpus = 'corpora/arabizi-english-bitext.en'
en_nltk_corpus = 'corpora/nltk_twitter_samples.csv'
#en_twitter_corpus = 'corpora/arab_politics-small.txt'
en_twitter_corpus = 'corpora/arab_politics_tweet.txt'


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

    if corpus == 'corpora/arabizi-twitter-egy.csv':
        sentence_df.loc[4064, 'language'] = 0
        sentence_df['language'] = sentence_df['language'].astype(int)
#    print(corpus, 'dtype:', sentence_df.language.dtype)
    
    sentence_df['sentence'] = sentence_df['sentence'].map(lambda x:preprocess(x))
    
    return sentence_df


def preprocess(sentence):
    """
    """
    remove_reply_tweet = re.sub(r'RT @', '@', sentence)
    shorten = re.sub(r'(.)\1+', r'\1\1', remove_reply_tweet)
    remove_misc = re.sub(r'(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', ' ', shorten)
    remove_punc = re.sub(r'/[!@#$%^&*()-=_+|;:",.<>?]/', ' ', remove_misc)
    remove_large_num = re.sub(r'[0-9]{3,}', ' ', remove_punc)
    remove_solo_num = re.sub(r' [0-9]+ ', ' ', remove_large_num)
    remove_solo_num = re.sub(r' [0-9]+ ', ' ', remove_solo_num)
    remove_extra_white = ' '.join(remove_solo_num.split())
    
    return remove_extra_white


def combine_sentence_df(sentence_df1, sentence_df2):
    """
    """
    sentence_df = sentence_df1.append(sentence_df2, ignore_index=True)
    
    return sentence_df


def combine_texts(text1, text2, text3, text4, text5, text6):
    """
    """    
    texta = combine_sentence_df(text1, text2)
    textb = combine_sentence_df(texta, text3)
    text = combine_sentence_df(textb, text4)   
    unseen_text = combine_sentence_df(text5, text6)
    print('text.shape:', text.shape)
    print('unseen_text.shape:', unseen_text.shape)
    
    return text, unseen_text
        

def count_vectorize(text, analyzer='char', n=3):
    """
    """
    print('analyzer:', analyzer, '<<<>>> n:', n)
    X_train, X_test, y_train, y_test = \
    train_test_split(text.sentence, text.language, test_size=0.1, \
                     random_state=0) 
    
    count_vectorizer = CountVectorizer(analyzer=analyzer, ngram_range=(n, n), \
                                       decode_error='ignore')
    count_train = count_vectorizer.fit_transform(X_train.values)
    count_test = count_vectorizer.transform(X_test.values)
        
    return count_vectorizer, count_train, count_test, y_train, y_test


def tfidf_vectorize(text, analyzer='char', n=3):
    """
    """
    print('analyzer:', analyzer, '<<<>>> n:', n) 
    X_train, X_test, y_train, y_test = \
    train_test_split(text.sentence, text.language, test_size=0.1, \
                     random_state=0)
    tfidf_vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=(n, n), \
                                       decode_error='ignore')
    
    tfidf_train = tfidf_vectorizer.fit_transform(X_train.values)
    tfidf_test = tfidf_vectorizer.transform(X_test.values)
    
    return tfidf_vectorizer, tfidf_train, tfidf_test, y_train, y_test


def vectorize_and_classify(text, unseen_text, analyzer='char', n=3):
    """
    """
    start = time.time()
    
    print('< count_vectorizer >')
    count_vectorizer, X_train, X_test, y_train, y_test = \
    count_vectorize(text, analyzer, n)
    
    nb_classifier, score, f1, confusion_mtrx = \
    train_naive_bayes(X_train, X_test, y_train, y_test)
    
    decode_naive_bayes(unseen_text, count_vectorizer, nb_classifier)
    
    print('>>> time taken:', (time.time()-start), '<<<\n')
    start = time.time()

    
    svm_classifier, score, f1, confusion_mtrx = \
    train_support_vector_machine(X_train, X_test, y_train, y_test)
    
    decode_support_vector_machine(unseen_text, count_vectorizer, svm_classifier)
    
    print('>>> time taken:', (time.time()-start), '<<<\n')
    start = time.time()

    print('< tfidf_vectorizer >')
    tfidf_vectorizer, X_train, X_test, y_train, y_test = \
    tfidf_vectorize(text, analyzer, n)
    
    nb_classifier, score, f1, confusion_mtrx = \
    train_naive_bayes(X_train, X_test, y_train, y_test)
    
    decode_naive_bayes(unseen_text, tfidf_vectorizer, nb_classifier)
    
    print('>>> time taken:', (time.time()-start), '<<<\n')
    start = time.time()
    
    svm_classifier, score, f1, confusion_mtrx = \
    train_support_vector_machine(X_train, X_test, y_train, y_test)
    
    decode_support_vector_machine(unseen_text, tfidf_vectorizer, svm_classifier)

    print('>>> time taken:', (time.time()-start), '<<<\n')
    
    return
 

def train_naive_bayes(X_train, X_test, y_train, y_test):
    """
    """
    print('* train naive bayes *')
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train, y_train)
    pred = nb_classifier.predict(X_test)
    
    score = metrics.accuracy_score(y_test, pred)
    print('accuracy:', score)
    
    f1 = metrics.f1_score(y_test, pred)#, average='weighted')
    print('f1:', f1)    
    
    confusion_mtrx = metrics.confusion_matrix(y_test, pred, labels=[1, 0])
    print(confusion_mtrx, '\n')

    return nb_classifier, score, f1, confusion_mtrx


def decode_naive_bayes(unseen_text, vectorizer, nb_classifier):
    """
    """
    print('* decode naive bayes *')
    y_test = unseen_text.language
    unseen_vectorizer = vectorizer.transform(unseen_text.sentence)
    pred = nb_classifier.predict(unseen_vectorizer)
    
    score = metrics.accuracy_score(y_test, pred)
    print('accuracy:', score)
    
    f1 = metrics.f1_score(y_test, pred)#, average='weighted')
    print('f1:', f1)    
    
    confusion_mtrx = metrics.confusion_matrix(y_test, pred, labels=[1, 0])
    print(confusion_mtrx, '\n')

    return nb_classifier, score, f1, confusion_mtrx


def train_support_vector_machine(X_train, X_test, y_train, y_test):
    """
    """
    print('* train support vector machine *')
    svm_classifier = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    svm_classifier.fit(X_train, y_train)
    pred = svm_classifier.predict(X_test)
    
    score = metrics.accuracy_score(y_test, pred)
    print('accuracy:', score)
    
    f1 = metrics.f1_score(y_test, pred)#, average='weighted')
    print('f1:', f1)
    
    confusion_mtrx = metrics.confusion_matrix(y_test, pred, labels=[1, 0])
    print(confusion_mtrx, '\n')
    
    return svm_classifier, score, f1, confusion_mtrx


def decode_support_vector_machine(unseen_text, vectorizer, svm_classifier):
    """
    """
    print('* decode support vector machine *')
    y_test = unseen_text.language
    unseen_vectorizer = vectorizer.transform(unseen_text.sentence)
    pred = svm_classifier.predict(unseen_vectorizer)
    
    score = metrics.accuracy_score(y_test, pred)
    print('accuracy:', score)
    
    f1 = metrics.f1_score(y_test, pred)#, average='weighted')
    print('f1:', f1)    
    
    confusion_mtrx = metrics.confusion_matrix(y_test, pred, labels=[1, 0])
    print(confusion_mtrx, '\n')

    return svm_classifier, score, f1, confusion_mtrx


def main():   
    """
    """
    start = time.time()
    print('* import and preprocessing *')
    ar_text = make_sentence_df(ar_corpus, '\n', header=None, target=1)
    ar_egy_text = make_sentence_df(ar_egy_corpus, ',', header=0)
    ar_leb_text = make_sentence_df(ar_leb_corpus, ',', header=0)
    en_text = make_sentence_df(en_corpus, '\n', header=None, target=0)
    en_nltk_text = make_sentence_df(en_nltk_corpus, ',', header=0, target=0)
    en_twitter_text = make_sentence_df(en_twitter_corpus, '\n', header=0, \
                                       target=0)
    text, unseen_text = combine_texts(ar_text, ar_egy_text, en_text, \
                                      en_nltk_text, \
                                      ar_leb_text, en_twitter_text)
    
    print('>>> time taken:', (time.time()-start), '<<<\n')
 
    vectorize_and_classify(text, unseen_text)
    vectorize_and_classify(text, unseen_text, analyzer='char', n=4)
    vectorize_and_classify(text, unseen_text, analyzer='char', n=5)
    vectorize_and_classify(text, unseen_text, analyzer='char', n=6)
    vectorize_and_classify(text, unseen_text, analyzer='word', n=1)
    vectorize_and_classify(text, unseen_text, analyzer='word', n=2)
    vectorize_and_classify(text, unseen_text, analyzer='word', n=3)
 
    print('>>> time taken:', (time.time()-start), '<<<\n')


if __name__=="__main__": 
    main() 