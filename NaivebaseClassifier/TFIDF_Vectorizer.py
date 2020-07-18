#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 12:59:16 2020

@author: muhammadsiraj
"""
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform([
    "Some people have curly brown hair till pointed black"
    ,"The quick brown fox jumps over the lazy dog"
    ,"black curly brown hair"
    ,"jumps over the lazy dog"
    ]
    )

#print(tfidf)
#print(tfidf.A)
print((tfidf*tfidf.T).A)


