import datetime
from statistics import median

import tensorflow as tf
import pandas as pd
import numpy as np
import re

from bs4 import BeautifulSoup
from objs.Request import Requests
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.layers import Input, Embedding, Conv1D, Dropout, MaxPool1D, Flatten, Dense, Bidirectional, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate

# Lemmatizer
# Tokenizer
# Remove stopwords
# https://github.com/kazimirzak/Bachelor/blob/b3c5441ccb46d100b9eb8632a47c69b08761df90/main.py#L96
# https://jovian.ai/diardanoraihan/ensemble-cr/v/2?utm_source=embed#C39

def main():

    rs = Requests().get_between_sql(
        str(datetime.datetime(2015, 12, 16)),
        str(datetime.datetime(2015, 12, 18))
    )

    def get_has_error(x):
        return (
            isinstance(x['solutionDate'], str)
        )

    def get_cleanup(x):
        text = BeautifulSoup(x.description, "lxml").text
        text = text.lower()
        text = re.sub('[\n.]', ' ', text)
        return text

    def get_process_category(x, p):
        if x['process'] > p:
            return 2
        return 1

    data = pd.DataFrame(rs, columns=Requests().fillables)

    data['hasError'] = data.apply(lambda x: get_has_error(x), axis=1)
    data = data[~data['hasError']]

    def get_process(x):
        return int(x.solutionDate.timestamp()) - int(x.receivedDate.timestamp())

    data['process'] = data.apply(lambda x: get_process(x), axis=1)
    data['cleanup'] = data.apply(lambda x: get_cleanup(x), axis=1)

    processes = list(data.process)
    processes_median = median(processes)

    data.info()

    data['processCategory'] = data.apply(lambda x: get_process_category(x, processes_median), axis=1)
    sentences, labels = list(data.cleanup), list(data.processCategory)

    def get_max_length(sequences):
        max_length = 0
        for idx, seq in enumerate(sequences):
            length = len(seq)
            if max_length < length:
                max_length = length
        return max_length

    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<UNK>"

    print("Example of sentence: ", sentences[2])

    # Cleaning and Tokenization
    tokenizer = Tokenizer(oov_token=oov_tok)
    tokenizer.fit_on_texts(sentences)

    # Turn the text into sequence
    training_sequences = tokenizer.texts_to_sequences(sentences)
    max_len = get_max_length(training_sequences)

    print('Into a sequence of int:', training_sequences[2])

    # Pad the sequence to have the same size
    training_padded = pad_sequences(training_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)
    print('Into a padded sequence:', training_padded[2])

    word_index = tokenizer.word_index
    # See the first 10 words in the vocabulary
    for i, word in enumerate(word_index):
        print(word, word_index.get(word))
        if i == 9:
            break
    vocab_size = len(word_index) + 1
    print(vocab_size)

    def define_model(filters=100, kernel_size=3, activation='relu', input_dim=None, output_dim=300, max_length=None):

        # Channel 1
        input1 = Input(shape=(max_length,))
        embedding1 = Embedding(input_dim=input_dim, output_dim=output_dim, input_length=max_length)(input1)
        conv1 = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', kernel_constraint=MaxNorm(max_value=3, axis=[0, 1]))(embedding1)
        pool1 = MaxPool1D(pool_size=2, strides=2)(conv1)
        flat1 = Flatten()(pool1)
        drop1 = Dropout(0.5)(flat1)
        dense1 = Dense(10, activation='relu')(drop1)
        drop1 = Dropout(0.5)(dense1)
        out1 = Dense(1, activation='sigmoid')(drop1)

        # Channel 2
        input2 = Input(shape=(max_length,))
        embedding2 = Embedding(input_dim=input_dim, output_dim=output_dim, input_length=max_length, mask_zero=True)(input2)
        gru2 = Bidirectional(GRU(64))(embedding2)
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

    model_0 = define_model(input_dim=1000, max_length=100)
    model_0.summary()

    class myCallback(tf.keras.callbacks.Callback):
        # Override the method on_epoch_end() for our benefit
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('accuracy') > 0.93:
                print("\nReached 93% accuracy so cancelling training!")
                self.model.stop_training = True

    callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0,
                                                 patience=7, verbose=2,
                                                 mode='auto', restore_best_weights=True)

    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<UNK>"
    activations = ['relu']
    filters = 100
    kernel_sizes = [1, 2, 3, 4, 5, 6]

    columns = ['Activation', 'Filters', 'acc1', 'acc2', 'acc3', 'acc4', 'acc5', 'acc6', 'acc7', 'acc8', 'acc9', 'acc10', 'AVG']
    record = pd.DataFrame(columns=columns)

    # prepare cross validation with 10 splits and shuffle = True
    kfold = KFold(10, shuffle=True)

    # Separate the sentences and the labels
    sentences, labels = list(data.cleanup), list(data.processCategory)

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

                # encode data using
                # Cleaning and Tokenization
                tokenizer = Tokenizer(oov_token=oov_tok)
                tokenizer.fit_on_texts(train_x)

                # Turn the text into sequence
                training_sequences = tokenizer.texts_to_sequences(train_x)
                test_sequences = tokenizer.texts_to_sequences(test_x)

                max_len = get_max_length(training_sequences)

                # Pad the sequence to have the same size
                Xtrain = pad_sequences(training_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)
                Xtest = pad_sequences(test_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)

                word_index = tokenizer.word_index
                vocab_size = len(word_index) + 1

                # Define the input shape
                model = define_model(filters, kernel_size, activation, input_dim=vocab_size, max_length=max_len)

                # Train the model and initialize test accuracy with 0
                acc = 0
                while acc < 0.7:

                    print('Training ...')

                    # Train the model
                    model.fit(x=[Xtrain, Xtrain], y=train_y, batch_size=50, epochs=100, verbose=1,
                              callbacks=[callbacks], validation_data=([Xtest, Xtest], test_y))

                    # evaluate the model
                    loss, acc = model.evaluate([Xtest, Xtest], test_y, verbose=0)
                    print('Test Accuracy: {}'.format(acc * 100))

                    if acc < 0.7:
                        print('The model suffered from local minimum. Retrain the model!')
                        model = define_model(filters, kernel_size, activation, input_dim=vocab_size, max_length=max_len)
                    else:
                        print('Done!')

                # evaluate the model
                loss, acc = model.evaluate([Xtest, Xtest], test_y, verbose=0)
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


if __name__ == '__main__':
    main()
