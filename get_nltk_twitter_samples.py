#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nltk.corpus import twitter_samples
from nltk.twitter import json2csv

    
input_file = twitter_samples.abspath("tweets.20150430-223406.json")

with open(input_file) as fp:
    json2csv(fp, 'tweets_text.csv', ['text'])