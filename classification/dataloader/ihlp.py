import datetime
import math
import os
import re
import warnings
from statistics import median

import lemmy
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk import word_tokenize

from objs.Request import Requests

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')


# Lemmatizer
# Tokenizer
# Remove stopwords
# MongoDB for text search
# https://github.com/kazimirzak/Bachelor/blob/b3c5441ccb46d100b9eb8632a47c69b08761df90/main.py#L96
# https://jovian.ai/diardanoraihan/ensemble-cr/v/2?utm_source=embed#C39
# https://github.com/miguelfzafra/Latest-News-Classifier/blob/master/0.%20Latest%20News%20Classifier/03.%20Feature%20Engineering/03.%20Feature%20Engineering.ipynb

class IHLP:

    @staticmethod
    def get_str_from_tokens(tokens):
        return " ".join(str(x) for x in tokens)

    @staticmethod
    def get_tokens_from_str(string):
        return string.split(" ")

    @staticmethod
    def get_stopwords_removed(tokens, stopwords=None):
        return [token for token in tokens if token not in stopwords]

    @staticmethod
    def get_lemma(lemmatizer, tokens):
        return [lemmatizer.lemmatize("", token)[0] for token in tokens]

    @staticmethod
    def get_tokenized_text(line, language="danish"):
        return [token.lower() for token in word_tokenize(line, language=language) if token.isalnum()]

    @staticmethod
    def get_beautiful_text(line):
        text = BeautifulSoup(line, "lxml").text
        text = re.sub('[\n.]', ' ', text)
        return text

    # --------------------------------------------------------------------------

    def ready_stop_words(
            self,
            language='danish',
            file_path_input='{}/input/stopwords.txt'.format(os.path.dirname(__file__)),
    ):
        print(os.path.dirname(__file__))

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
            self,
            file_path_input='{}/input/data.xlsx'.format(os.path.dirname(__file__))
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

        def get_process_time(x):
            x = int(x.dateStart.timestamp()) - int(x.dateEnd.timestamp())
            if x < 1:
                return 0
            return np.log(x) / np.log(10)

        def get_process_text(text):
            text = IHLP.get_beautiful_text(text)
            tokens = IHLP.get_tokenized_text(text)
            tokens = IHLP.get_lemma(tokens=tokens, lemmatizer=lemmatizer)
            tokens = IHLP.get_stopwords_removed(tokens=tokens, stopwords=stopwords)
            return IHLP.get_str_from_tokens(tokens)

        stopwords = self.ready_stop_words()
        lemmatizer = lemmy.load("da")

        rs = Requests().get_between_sql(str_from, str_to)

        data = pd.DataFrame(rs, columns=Requests().fillables)

        data['dateStart'] = data.apply(lambda x: get_date(x, index='solutionDate'), axis=1)
        data = data[~data.dateStart.isnull()]

        data['dateEnd'] = data.apply(lambda x: get_date(x, index='receivedDate'), axis=1)
        data = data[~data.dateEnd.isnull()]

        data['processTime'] = data.apply(lambda x: get_process_time(x), axis=1)
        data['processText'] = data.apply(lambda x: get_process_text(
            "{} {}".format(x['subject'], x['description'])
        ), axis=1)

        data['processCategory'] = pd.qcut(data['processTime'],
                                          q=[0, .2, .4, .6, .8, 1],
                                          labels=[1, 2, 3, 4, 5])

        data = data[['processTime', 'processText', 'processCategory']]
        data.to_excel('input/data.xlsx', index=False)

        return self.ready_data()

    def get(self):
        data = self.ready_data()
        data = data.fillna('')
        data = data[data['processText'].str.split().str.len().lt(300)]
        data = data[data['processTime'] >= 0]
        return data[['processText', 'processCategory']]
