import config

from dataloader.loader import Loader
from preprocessing.utils import Preprocess, remove_empty_docs
from dataloader.embeddings import GloVe
from model.cnn_document_model import DocumentModel, TrainingParameters
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from utils import scatter_plot

import numpy as np
import keras.backend as kb

from sklearn.manifold import TSNE

dataset = Loader.load_20newsgroup_data(subset='train')
corpus, labels = dataset.data, dataset.target
corpus, labels = remove_empty_docs(corpus, labels)

test_dataset = Loader.load_20newsgroup_data(subset='test')
test_corpus, test_labels = test_dataset.data, test_dataset.target
test_corpus, test_labels = remove_empty_docs(test_corpus, test_labels)

six_groups = {
    'comp.graphics': 0,
    'comp.os.ms-windows.misc': 0,
    'comp.sys.ibm.pc.hardware': 0,
    'comp.sys.mac.hardware': 0,
    'comp.windows.x': 0,
    'rec.autos': 1,
    'rec.motorcycles': 1,
    'rec.sport.baseball': 1,
    'rec.sport.hockey': 1,
    'sci.crypt': 2,
    'sci.electronics': 2,
    'sci.med': 2,
    'sci.space': 2,
    'misc.forsale': 3,
    'talk.politics.misc': 4,
    'talk.politics.guns': 4,
    'talk.politics.mideast': 4,
    'talk.religion.misc': 5,
    'alt.atheism': 5,
    'soc.religion.christian': 5
}

map_20_2_6 = [six_groups[dataset.target_names[i]] for i in range(20)]
labels = [six_groups[dataset.target_names[i]] for i in labels]
test_labels = [six_groups[dataset.target_names[i]] for i in test_labels]

Preprocess.MIN_WD_COUNT = 5
preprocessor = Preprocess(corpus=corpus)
corpus_to_seq = preprocessor.fit()

test_corpus_to_seq = preprocessor.transform(test_corpus)

glove = GloVe(50)
initial_embeddings = glove.get_embedding(preprocessor.word_index)

newsgroup_model = DocumentModel(vocab_size=preprocessor.get_vocab_size(),
                                sent_k_maxpool=5,
                                sent_filters=20,
                                word_kernel_size=5,
                                word_index=preprocessor.word_index,
                                num_sentences=Preprocess.NUM_SENTENCES,
                                embedding_weights=initial_embeddings,
                                conv_activation='relu',
                                train_embedding=True,
                                learn_word_conv=True,
                                learn_sent_conv=True,
                                sent_dropout=0.4,
                                hidden_dims=64,
                                input_dropout=0.2,
                                hidden_gaussian_noise_sd=0.5,
                                final_layer_kernel_regularizer=0.1,
                                num_hidden_layers=2,
                                num_units_final_layer=6)

train_params = TrainingParameters('6_newsgrp_largeclass',
                                  model_file_path=config.MODEL_DIR + '/20newsgroup/model_6_01.hdf5',
                                  model_hyper_parameters=config.MODEL_DIR + '/20newsgroup/model_6_01.json',
                                  model_train_parameters=config.MODEL_DIR + '/20newsgroup/model_6_01_meta.json',
                                  num_epochs=20,
                                  batch_size=128,
                                  validation_split=.10,
                                  learning_rate=0.01)

train_params.save()
newsgroup_model.save_model(train_params.model_hyper_parameters)
newsgroup_model.model.compile(loss="categorical_crossentropy", optimizer=train_params.optimizer, metrics=["accuracy"])
checkpointer = ModelCheckpoint(filepath=train_params.model_file_path, verbose=1, save_best_only=True,
                               save_weights_only=True)

early_stop = EarlyStopping(patience=2)

x_train = np.array(corpus_to_seq)
y_train = to_categorical(np.array(labels))

x_test = np.array(test_corpus_to_seq)
y_test = to_categorical(np.array(test_labels))

# Set LR
kb.set_value(newsgroup_model.get_classification_model().optimizer.lr, train_params.learning_rate)

newsgroup_model.get_classification_model().fit(x_train, y_train,
                                               batch_size=train_params.batch_size,
                                               epochs=train_params.num_epochs,
                                               verbose=2,
                                               validation_split=train_params.validation_split,
                                               callbacks=[checkpointer, early_stop])

newsgroup_model.get_classification_model().evaluate(x_test, y_test, verbose=2)
preds = newsgroup_model.get_classification_model().predict(x_test)
preds_test = np.argmax(preds, axis=1)

print(classification_report(test_labels, preds_test))
print(confusion_matrix(test_labels, preds_test))
print(accuracy_score(test_labels, preds_test))

doc_embeddings = newsgroup_model.get_document_model().predict(x_test)
print(doc_embeddings.shape)

doc_proj = TSNE(n_components=2, random_state=42, ).fit_transform(doc_embeddings)

f, ax, sc, txts = scatter_plot(doc_proj, np.array(test_labels))
f.savefig('nws_grp_embd.png')
