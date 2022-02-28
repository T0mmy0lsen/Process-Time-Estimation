### Goals
- Problem ticket assignment
- Auto resolution of problem tickets
- Scheduling og the tickets
### Text categorization
- Sentence splitting
- Tokenization
- Stemming or lemmatization
- Text cleanup
The following are some popular text classification algorithms:
- Multinomial Naive Bayes
- Support Vector Machine
- k-Nearest Neighbor
### Bag of Words
Feature representation of a document. 
Choose N words, w is the frequency of the Nth word,
then BoW is D = { w1,w2, ... wn }

Shortcomings of BoW models
With the word count-based BoW model, we lose additional information such as the
semantics, structure, sequence, and context around nearby words in each text document.
Words with similar meaning are treated differently in BoW

It's observed that, in LSI models,
words with similar semantics have close representations. Also, this dense representation of
words is the first step for applying deep learning models to text and is called word
embedding. Neural network-based language models try to predict words from their
neighboring words by looking at word sequences in the corpus, and in the process learn
distributed representations, giving us dense word embedding.

***
**TLWP p.240**\
This representation of text as a sparse vector is called the BoW model. Here, we don't consider
the sequential nature of text data. One way to partially capture the sequential information
is to consider word *phrases or n-grams* along with single word features while building the
vocabulary. However, one challenge with that is the size of our representation; that is, our
vocabulary size explodes.
***
### TF-IDF
The most popular
representation is a normalized representation of the word frequency called Term
Frequency-Inverse Document Frequency (TF-IDF) representation.

*Others*
- Binary
- IDF (Inverse document frequency)

It increases proportionally to the number of times a word appears in the
document and is scaled down by the frequency of the word in the corpus, which helps to
adjust for the fact that some words appear more frequently in general.

# Word2Vec model
### Continuous Bag of Words 
Given a text window, we try to predict the target (next) word.
The Word2vec family of models are unsupervised; this means that you can just give it a corpus without additional
labels or information and it can construct dense word embeddings from the corpus. 
