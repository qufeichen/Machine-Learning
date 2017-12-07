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

# First Test: querying two words:
input_1 = 'cat'
input_2 = 'kitten'

result_1 = model.similarity(input_1, input_2)

# Second Test: querying two lists:
input_list_1 = ['cats', 'are', 'inferior', 'than', 'dogs']
input_list_2 = ['dogs', 'are', 'way', 'better', 'than', 'cats']

result_2 = model.n_similarity(input_list_1, input_list_2)

# Second Test: Testing positive and negative
input_3 = ['pet', 'young', 'dog']
input_4 = ['cat']

result_3 = model.most_similar(positive=input_3, negative=input_4)

# end timer
stop_time = timeit.default_timer()

# print results
print('Similarity between {} and {}:\n {}'.format(input_1, input_2, result_1))
print('Similarity between {} and {}:\n {}'.format(input_list_1, input_list_2, result_2))
print('Most Similar with Positive: {}, Negative: {}:\n {}'.format(input_3, input_4, result_3))
# print times
print('Time taken to load model: {}'.format(start_time - model_load_time))
print('Total Execution time: {}\n'.format(start_time - stop_time))
