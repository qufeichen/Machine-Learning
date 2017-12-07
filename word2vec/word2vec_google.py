# simple query of google's word2vec model
# google's pretrained model was obtained here: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
import gensim
import os
import string
import re
from sklearn.feature_extraction import text
import collections

# Load Google's pre-trained Word2Vec model.
print('\nLoading Model...')
print('---------------------------------------------------\n')
model = gensim.models.KeyedVectors.load_word2vec_format('/Users/qufeichen/Documents/Repos/Machine-Learning/word2vec/GoogleNews-vectors-negative300.bin.gz', binary=True)

# run query
result = model.similarity('cat', 'kitten')
print(result)
