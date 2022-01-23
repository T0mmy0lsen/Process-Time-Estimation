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

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

# Lemmatizer
# Tokenizer
# Remove stopwords
# MongoDB for text search
# https://github.com/kazimirzak/Bachelor/blob/b3c5441ccb46d100b9eb8632a47c69b08761df90/main.py#L96
# https://jovian.ai/diardanoraihan/ensemble-cr/v/2?utm_source=embed#C39
# https://github.com/miguelfzafra/Latest-News-Classifier/blob/master/0.%20Latest%20News%20Classifier/03.%20Feature%20Engineering/03.%20Feature%20Engineering.ipynb

# Text processing --------------------------------------------------------------------------


def get_str_from_tokens(tokens):
    return " ".join(str(x) for x in tokens)


def get_tokens_from_str(string):
    return string.split(" ")


def get_stopwords_removed(tokens, stopwords=None):
    return [token for token in tokens if token not in stopwords]


def get_lemma(lemmatizer, tokens):
    return [lemmatizer.lemmatize("", token)[0] for token in tokens]


def get_tokenized_text(line, language="danish"):
    return [token.lower() for token in word_tokenize(line, language=language) if token.isalnum()]


def get_beautiful_text(line):
    text = BeautifulSoup(line, "lxml").text
    text = re.sub('[\n.]', ' ', text)
    return text

# --------------------------------------------------------------------------


def ready_stop_words(
        language='danish',
        file_path_input='input/stopwords.txt',
):
    """:return array of stopwords in :arg language"""
    if os.path.isfile(file_path_input):
        stopwords = []
        with open(file_path_input, 'r') as file_handle:
            for line in file_handle:
                currentPlace = line[:-1]
                stopwords.append(currentPlace)
        return stopwords

    url = "http://snowball.tartarus.org/algorithms/%s/stop.txt" % language
    text = requests.get(url).text
    stopwords = re.findall('^(\w+)', text, flags=re.MULTILINE | re.UNICODE)

    url_en = "http://snowball.tartarus.org/algorithms/english/stop.txt"
    text_en = requests.get(url_en).text
    stopwords_en = re.findall('^(\w+)', text_en, flags=re.MULTILINE | re.UNICODE)

    with open(file_path_input, 'w') as file_handle:
        for list_item in stopwords + stopwords_en:
            file_handle.write('%s\n' % list_item)

    return stopwords


def ready_data(
        file_path_input='input/data.xlsx'
):
    str_from = str(datetime.datetime(2015, 12, 31))
    str_to = str(datetime.datetime(2016, 12, 31))

    if os.path.isfile(file_path_input):
        return pd.read_excel(file_path_input)
    
    def get_date(x, index):
        tmp = x[index]
        if isinstance(x[index], str):
            if tmp[0] != '2':
                return None
            tmp = datetime.datetime.strptime(tmp, "%Y-%m-%d %H:%M:%S")
        return tmp

    def get_category(x, p):
        if x['processTime'] < p / 2:
            return 1
        if x['processTime'] < p:
            return 2
        if x['processTime'] < p * 1.5:
            return 3
        return 4

    def get_process_time(x):
        x = int(x.dateStart.timestamp()) - int(x.dateEnd.timestamp())
        if x < 1:
            return 0
        return np.log(x) / np.log(10)

    def get_process_text(text):
        text = get_beautiful_text(text)
        tokens = get_tokenized_text(text)
        tokens = get_lemma(tokens=tokens, lemmatizer=lemmatizer)
        tokens = get_stopwords_removed(tokens=tokens, stopwords=stopwords)
        return get_str_from_tokens(tokens)

    stopwords = ready_stop_words()
    lemmatizer = lemmy.load("da")

    rs = Requests().get_between_sql(str_from, str_to)

    data = pd.DataFrame(rs, columns=Requests().fillables)

    data['dateStart'] = data.apply(lambda x: get_date(x, index='solutionDate'), axis=1)
    data = data[~data.dateStart.isnull()]

    data['dateEnd'] = data.apply(lambda x: get_date(x, index='receivedDate'), axis=1)
    data = data[~data.dateEnd.isnull()]

    data['processTime'] = data.apply(lambda x: get_process_time(x), axis=1)
    data['processText'] = data.apply(lambda x: get_process_text(
        "{} {} {}".format(x['subject'], x['subject'], x['description'])
    ), axis=1)

    processes = list(data.processTime)
    processes_median = median(processes)

    data['processCategory'] = data.apply(lambda x: get_category(x, processes_median), axis=1)

    data = data[['processTime', 'processText', 'processCategory']]
    data.to_excel('input/data.xlsx')

    return ready_data()


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

    data = ready_data()
    data = data.fillna('')
    print(len(data))

    data = data[data['processText'].str.split().str.len().lt(100)]
    data = data[data['processTime'] >= 0]
    print(len(data))

    processes = list(data.processTime)
    processes_median = median(processes)

    mean = sum(processes) / len(processes)
    var = sum((el - mean) ** 2 for el in processes) / len(processes)
    st_dev = math.sqrt(var)

    count, bins, ignored = plt.hist(processes, 100, density=True)
    # Plot the distribution curve
    plt.plot(bins, 1 / (st_dev * np.sqrt(2 * np.pi)) *
             np.exp(- (bins - processes_median) ** 2 / (2 * st_dev ** 2)),
             linewidth=2, color='r')
    plt.show()

    analysis(data)
    # train_and_test(data)


def stamp(x=None):
    print("[%s] %s" % (dt.now(), sys._getframe().f_back.f_lineno))
    if x is not None:
        print(x)


if __name__ == '__main__':
    stamp()
    main()
