# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 15:34:14 2018

@author: tghosh
"""

import config
from dataloader.loader import Loader
from preprocessing.utils import Preprocess, remove_empty_docs
from dataloader.embeddings import GloVe
from model.cnn_document_model import DocumentModel, TrainingParameters
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np

train_df = Loader.load_ihlp()
print(train_df.shape)

test_df = Loader.load_ihlp()
print(test_df.shape)

dataset = train_df.sample(n=20000, random_state=42)
dataset.processCategory.value_counts()

corpus = dataset['processText'].values
target = dataset['processCategory'].values
print(corpus.shape, target.shape)

corpus, target = remove_empty_docs(corpus, target)
print(len(corpus))

preprocessor = Preprocess(corpus=corpus)
corpus_to_seq = preprocessor.fit()

holdout = train_df.sample(n=10000, random_state=52)
holdout_corpus = holdout['processText'].values
holdout_target = holdout['processCategory'].values
print(holdout_corpus.shape, holdout_target.shape)

holdout_corpus, holdout_target = remove_empty_docs(holdout_corpus, holdout_target)
print(len(holdout_corpus))
holdout_corpus_to_seq = preprocessor.transform(holdout_corpus)

glove = GloVe(50)
initial_embeddings = glove.get_embedding(preprocessor.word_index)

amazon_review_model = DocumentModel(vocab_size=preprocessor.get_vocab_size(),
                                    word_index=preprocessor.word_index,
                                    num_sentences=Preprocess.NUM_SENTENCES,
                                    embedding_dim=50,
                                    # embedding_weights=initial_embeddings,
                                    embedding_regularizer_l2=0.1,
                                    sentence_len=30,
                                    word_filters=30,
                                    sent_filters=16,
                                    doc_k_maxpool=5,
                                    conv_activation='relu',
                                    train_embedding=False,
                                    learn_word_conv=True,
                                    learn_sent_conv=True,
                                    hidden_dims=16,
                                    num_hidden_layers=2,
                                    sent_dropout=0.1,
                                    input_dropout=0.05,
                                    hidden_gaussian_noise_sd=0.5,
                                    final_layer_kernel_regularizer=0.04)

train_params = TrainingParameters('model_with_tanh_activation',
                                  model_file_path=config.MODEL_DIR + '/ihlp/model_06.hdf5',
                                  model_hyper_parameters=config.MODEL_DIR + '/ihlp/model_06.json',
                                  model_train_parameters=config.MODEL_DIR + '/ihlp/model_06_meta.json',
                                  num_epochs=30, batch_size=64)

train_params.save()

amazon_review_model.model.compile(loss="binary_crossentropy", optimizer=train_params.optimizer, metrics=["accuracy"])
checkpointer = ModelCheckpoint(filepath=train_params.model_file_path, verbose=1, save_best_only=True,
                               save_weights_only=True)

early_stop = EarlyStopping(patience=2)

x_train = np.array(corpus_to_seq)
y_train = np.array(target)

x_test = np.array(holdout_corpus_to_seq)
y_test = np.array(holdout_target)

print(x_train.shape, y_train.shape)

amazon_review_model.get_classification_model().fit(x_train, y_train,
                                                   batch_size=train_params.batch_size,
                                                   epochs=train_params.num_epochs,
                                                   verbose=2,
                                                   validation_split=train_params.validation_split,
                                                   callbacks=[checkpointer])

amazon_review_model.get_classification_model().evaluate(x_test, y_test, train_params.batch_size * 10, verbose=2)
amazon_review_model.save_model(train_params.model_hyper_parameters)

# Which embeddings changes most

learned_embeddings = amazon_review_model.get_classification_model().get_layer('imdb_embedding').get_weights()[0]

embd_change = {}
for word, i in preprocessor.word_index.items():
    embd_change[word] = np.linalg.norm(initial_embeddings[i] - learned_embeddings[i])
embd_change = sorted(embd_change.items(), key=lambda x: x[1], reverse=True)
