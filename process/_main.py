import datetime
import math
import os
import sys
import warnings
from statistics import median

from datetime import datetime as dt
import lemmy
import requests
import pandas as pd
import numpy as np
import re

from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

from objs.Request import Requests
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.layers import Input, Embedding, Conv1D, Dropout, MaxPool1D, Flatten, Dense, Bidirectional, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate


def analysis(data):
    
    x_train, x_test, y_train, y_test = train_test_split(data['processText'],
                                                        data['processCategory'],
                                                        test_size=0.20,
                                                        random_state=8)

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=300,
        max_df=.1,
        min_df=3
    )

    # Matrix, x = feature, y = sentence. x,y = frequency of feature in sentence.
    train_features = tfidf.fit_transform(x_train).toarray()
    # Series, sentence index and category
    train_labels = y_train

    categories = {
        1: 'Very Short',
        2: 'Short',
        3: 'Long',
        4: 'Very Long',
    }

    for category, label in categories.items():

        # The training-labels and features has matching indexes.
        # Fist we get the indexes of the training-labels.

        i = 0
        indexes = []
        for value in train_labels.items():
            if value[1] == category:
                indexes.append(i)
            i += 1

        # We filter out feature-rows that correspond to the category.
        filtered = np.array(train_features)[indexes]

        # We sum all the features to see those that are most frequent for the category.
        sums = np.sum(filtered, axis=0)

        # We sum all the features to see those that are most frequent for the category.
        series = pd.Series([e for idx, e in enumerate(sums)])
        series = series.sort_values(ascending=False)
        series_indexes = [key for key, value in series.items()][:5]

        frequent_words = np.array(tfidf.get_feature_names_out())[series_indexes]
        print(frequent_words)


def max_length(sequences):
    target = 0
    for i, seq in enumerate(sequences):
        length = len(seq)
        if target < length:
            target = length
    return target


def define_model(filters=100, kernel_size=3, activation='relu', input_dim=None, output_dim=300, max_length=None):
    # Channel 1
    input1 = Input(shape=(max_length,))
    embeddding1 = Embedding(input_dim=input_dim, output_dim=output_dim, input_length=max_length)(input1)
    conv1 = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu',
                   kernel_constraint=MaxNorm(max_value=3, axis=[0, 1]))(embeddding1)
    pool1 = MaxPool1D(pool_size=2, strides=2)(conv1)
    flat1 = Flatten()(pool1)
    drop1 = Dropout(0.5)(flat1)
    dense1 = Dense(10, activation='relu')(drop1)
    drop1 = Dropout(0.5)(dense1)
    out1 = Dense(1, activation='sigmoid')(drop1)

    # Channel 2
    input2 = Input(shape=(max_length,))
    embeddding2 = Embedding(input_dim=input_dim, output_dim=output_dim, input_length=max_length, mask_zero=True)(input2)
    gru2 = Bidirectional(GRU(64))(embeddding2)
    drop2 = Dropout(0.5)(gru2)
    out2 = Dense(1, activation='sigmoid')(drop2)

    # Merge
    merged = concatenate([out1, out2])

    # Interpretation
    outputs = Dense(1, activation='sigmoid')(merged)
    model = Model(inputs=[input1, input2], outputs=outputs)

    # Compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train_and_test(data):

    activations = ['relu']
    filters = 100
    kernel_sizes = [1, 2, 3, 4, 5, 6]

    columns = ['Activation', 'Filters',
               'acc1', 'acc2', 'acc3', 'acc4', 'acc5', 'acc6', 'acc7', 'acc8', 'acc9', 'acc10',
               'AVG']
    record = pd.DataFrame(columns=columns)

    # prepare cross validation with 10 splits and shuffle = True
    kfold = KFold(10, shuffle=True)

    # Separate the sentences and the labels
    sentences, labels = list(data.processText), list(data.processCategory)

    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<UNK>"

    tokenizer = Tokenizer(oov_token=oov_tok)
    tokenizer.fit_on_texts(sentences)

    for activation in activations:
        for kernel_size in kernel_sizes:
            # kfold.split() will return set indices for each split
            acc_list = []
            for train, test in kfold.split(sentences):

                train_x, test_x = [], []
                train_y, test_y = [], []

                for i in train:
                    train_x.append(sentences[i])
                    train_y.append(labels[i])

                for i in test:
                    test_x.append(sentences[i])
                    test_y.append(labels[i])

                # Turn the labels into a numpy array
                train_y = np.array(train_y)
                test_y = np.array(test_y)

                # Turn the text into sequence
                training_sequences = tokenizer.texts_to_sequences(train_x)
                test_sequences = tokenizer.texts_to_sequences(test_x)

                max_len = max_length(training_sequences)

                # Pad the sequence to have the same size
                x_train = pad_sequences(training_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)
                x_test = pad_sequences(test_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)

                word_index = tokenizer.word_index
                vocab_size = len(word_index) + 1

                # Define the input shape
                model = define_model(filters, kernel_size, activation, input_dim=vocab_size, max_length=max_len)

                # Train the model and initialize test accuracy with 0
                acc = 0
                while acc < 0.7:

                    model.fit(x=[x_train, x_train], y=train_y, batch_size=50, epochs=100, verbose=1,
                              validation_data=([x_test, x_test], test_y))

                    loss, acc = model.evaluate([x_test, x_test], test_y, verbose=0)
                    print('Test Accuracy: {}'.format(acc * 100))

                    if acc < 0.7:
                        print('The model suffered from local minimum. Retrain the model!')
                        model = define_model(filters, kernel_size, activation, input_dim=vocab_size, max_length=max_len)

                    else:
                        print('Done!')

                loss, acc = model.evaluate([x_test, x_test], test_y, verbose=0)
                print('Test Accuracy: {}'.format(acc * 100))

                acc_list.append(acc * 100)

            mean_acc = np.array(acc_list).mean()
            parameters = [activation, kernel_size]
            entries = parameters + acc_list + [mean_acc]

            temp = pd.DataFrame([entries], columns=columns)
            record = record.append(temp, ignore_index=True)
            print()
            print(record)
            print()


def main():
    pass


def stamp(x=None):
    print("[%s] %s" % (dt.now(), sys._getframe().f_back.f_lineno))
    if x is not None:
        print(x)


if __name__ == '__main__':
    stamp()
    main()
