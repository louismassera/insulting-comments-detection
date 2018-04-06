import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.sparse import *
from scipy.sparse.linalg import *
from scipy.io import mmwrite
from nltk.stem import WordNetLemmatizer
import re

def get_vocab(corpus,n_grams):
    """ get vocabulary of the corpus

    parameters
    -----
    corpus : array (n_samples,) of cleaned documents

    return
    -----
    vocabulary : array (n_words,) of words
    """
    vocabulary = []
    grams_2 = []
    grams_3 = []
    wnl = WordNetLemmatizer()
    for doc in corpus:
        words = doc.split(" ")
        for word in words :
            string = str(wnl.lemmatize(word))
            string = str(wnl.lemmatize(string,'v'))
            if(string not in vocabulary):
                vocabulary.append(string)

        if (n_grams > 1):
            for i in range(len(words)-2):
                string = " ".join(words[i:i+2])
                string = str(wnl.lemmatize(string))
                string = str(wnl.lemmatize(string,'v'))
                if(string not in grams_2):
                    grams_2.append(string)
        if (n_grams > 2):
            for i in range(len(words)-3):
                string = " ".join(words[i:i+3])
                string = str(wnl.lemmatize(string))
                string = str(wnl.lemmatize(string,'v'))
                if(string not in grams_3):
                    grams_3.append(string)

    if(n_grams ==2):
        output = np.r_[np.asarray(vocabulary),np.asarray(grams_2)]
    elif(n_grams >2):
        output = np.r_[np.asarray(vocabulary),np.asarray(grams_2),np.asarray(grams_3)]
    else:
        output = np.asarray(vocabulary)
    output.sort()
    return output

def count_words(corpus,vocabulary,n_grams):
    """ get counts of words in dictionnary for each doc of a corpus

    parameters
    -----
    corpus : array (n_samples,) of cleaned documents

    return
    -----
    counts : array (n_samples,n_words) of words counts for each document in corpus
    """
    n_samples = corpus.shape[0]
    n_words = vocabulary.shape[0]
    wnl = WordNetLemmatizer()
    row = []
    col = []
    data = []
    for i, doc in enumerate(corpus):

        doc_splits = doc.split(" ")
        doc_2splits = []
        doc_3splits = []
        if(n_grams > 1):
            for n in range(len(doc_splits)-2):
                string = " ".join(doc_splits[n:n+2])
                string = str(wnl.lemmatize(string))
                string = str(wnl.lemmatize(string,'v'))
                doc_2splits.append(string)
        if(n_grams > 2):
            for n in range(len(doc_splits)-3):
                string = " ".join(doc_splits[n:n+3])
                string = str(wnl.lemmatize(string))
                string = str(wnl.lemmatize(string,'v'))
                doc_3splits.append(string)
        if(n_grams > 1):
            doc_splits = np.r_[np.asarray(doc_splits),np.asarray(doc_2splits)]
        elif(n_grams > 3):
            doc_splits = np.r_[np.asarray(doc_splits),np.asarray(doc_2splits),np.asarray(doc_3splits)]
        unique, freq = np.unique(doc_splits,return_counts=True)
        voc_doc = {}
        for (v,f) in zip(unique,freq):
            v = str(wnl.lemmatize(v))
            v = str(wnl.lemmatize(v,'v'))
            voc_doc.setdefault(v,f)

        for j, word in enumerate(vocabulary):
            if(word in voc_doc):
                data.append(voc_doc[word])
                row.append(i)
                col.append(j)

    counts = csc_matrix((data,(row,col)),shape=(n_samples,n_words))
    #mmwrite('conts_sparse',counts)
    return counts

def tfidf(counts):
    counts = csc_matrix(counts)
    n_samples = counts.shape[0]
    n_words = counts.shape[1]
    row = []
    col = []
    data = []
    counts = counts.tocoo()
    r_indices = counts.row
    c_indices = counts.col
    counts = counts.tocsc()
    tf = 1. / np.asarray(counts.sum(axis=1)).reshape(-1)
    for (i,j) in zip(r_indices,c_indices):
        idf = np.log(float(n_samples) / counts[:,j].nnz)
        row.append(i)
        col.append(j)
        data.append(tf[i]* counts[i,j] * (1 + idf))

    tfidf = csc_matrix((data,(row,col)),shape=(n_samples,n_words))
    tfidf = csc_matrix((tfidf.T / norm(tfidf,axis=1)).T)
    #mmwrite('tfidf_sparse',tfidf)
    return tfidf.toarray()


def clean_corpus(train_fname='data/train.csv',train_prepared_fname = 'data/train_prepared.csv',test_fname='data/test.csv'):

    """ clean the documents from unwanted chars
    parameters
    -----
    corpus : array (n_samples,) of cleaned documents

    return
    -----
    X_clean : dataiku cleaned array of comments
    X_train : python processed array of comments of training set
    y_train : labels
    X_test : python processed array of comments of test set
    """
    X_clean = []
    df_clean = pd.DataFrame(pd.read_csv(train_prepared_fname)['col_1'])
    X_clean = df_clean.values.reshape(df_clean.values.shape[0],)

    X_train = []
    y_train = []

    with open(train_fname) as f:
        for line in f:
            y_train.append(int(line[0]))
            l = line[5:-6]
            l = l.lower()
            #l = re.sub(r"f[u\*ck]* ","fuck ",l)
            #l = re.sub("_"," ",l)
            l = re.sub("\."," ",l)
            l = re.sub(r"http\S+"," ",l) #URLs
            l = re.sub(r"www\S+"," ",l) #URLs
            l = re.sub(r"<[^>]+>",' ',l) #HTML
            l = re.sub(r"[\"\\']",' ',l)
            l = re.sub(r"[=~\+\^&%*µ$£!§:;\.,\?#@<>\(\)\{\}\[\]\/\\\-]","",l) #weird stuff
            l = re.sub(r"x[a-z][0-9]"," ",l) #exa chars
            l = l.replace(r" [sdpnxto] {1}",' ') #smiley or stop words
            l = re.sub(r"[0-9]+\w+",' ',l)
            X_train.append(l)

    X_test = []
    with open(test_fname) as f:
        for line in f:
            l = line[3:-6]
            l = l.lower()
            #l = re.sub(r"f[u\*ck]* ","fuck ",l)
            #l = re.sub("_"," ",l)
            l = re.sub("\."," ",l)
            l = re.sub(r"http\S+"," ",l) #URLs
            l = re.sub(r"www\S+"," ",l) #URLs
            l = re.sub(r"<[^>]+>",' ',l) #HTML
            l = re.sub(r"[\"\\']",' ',l)
            l = re.sub(r"[=~\+\^&%*µ$£!§:;\.,\?#@<>\(\)\{\}\[\]\/\\\-]",'',l) #weird stuff
            l = re.sub(r"x[a-z][0-9]"," ",l) #exa chars
            l = l.replace(r" [sdpnxto] {1}",' ') #smiley or stop words
            l = re.sub(r"[0-9]+\w+",' ',l)
            X_test.append(l)

    y_train = np.array(y_train)
    y_train = 2*y_train -1
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    return X_clean,X_train,y_train,X_test

def TfidfVectorizerTransform(X_1,X_2,n_grams):
    """
    Parameters
    -----
    X_1 : array shape:(n_samples_1,) of comments which makes the dictionnary
    X_2 : array shape;(n_samples,) to vectorize and tfidf transform

    Return
    -----
    vocabulary : array shape:(n_words,) of ngrams from X_1
    counts : sparse shape:(n_samples,n_words) of words counts in X_2
    X_train: array shape:(n_samples,n_words) tfidf of counts
    """
    vocabulary = get_vocab(X_1,n_grams)
    counts = count_words(X_2,vocabulary,n_grams)
    X_train = tfidf(counts)
    #pd.DataFrame(data=X_train).to_csv('X_tfidf.csv')
    return vocabulary, counts, X_train
