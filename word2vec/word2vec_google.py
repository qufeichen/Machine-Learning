# simple query of google's word2vec model
# google's pretrained model was obtained here: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
import gensim
import os
import string
import re
from sklearn.feature_extraction import text
import collections
import timeit

# start timer - used to record run time
start_time = timeit.default_timer()

# Load Google's pre-trained Word2Vec model.
print('\nLoading Model...')
print('---------------------------------------------------')
model = gensim.models.KeyedVectors.load_word2vec_format('/Users/qufeichen/Documents/Repos/Machine-Learning/word2vec/GoogleNews-vectors-negative300.bin.gz', binary=True)

# timer for model load time
model_load_time = timeit.default_timer()

# query fields
input1 = 'cat'
input2 = 'kitten'


# run query
result = model.similarity(input1, input2)

# time to load model
stop_time = timeit.default_timer()

# print results
print('Similarity between {} and {}: {}'.format(input1, input2, result))
# print times
print('Time taken to load model: {}'.format(start_time - model_load_time))
print('Total Execution time: {}\n'.format(start_time - stop_time))
