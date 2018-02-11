#-*- coding: utf-8 -*-
# author: shibei00
# email: bshi@se.cuhk.edu.hk

vocabulary_size = 10000 # the size of the vocabulary
embedding_dimension = 200
max_length = 100 # the maximum size of the sentences
min_count = 5
display_steps = 200
lam = 0.5
batch_size = 32 # min value is 16 because sentiment
learning_rate = 1
num_skips = 2
skip_window = 1
num_sampled = 5 # the number of negative samples
# num_steps = 20000
num_steps = 200000
e_steps = 10000

# validation set
valid_size = 16
valid_window = 100
