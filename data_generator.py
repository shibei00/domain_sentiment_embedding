#-*- coding:utf-8 -*-
# author: shibei00
# email: bshi@se.cuhk.edu.hk

import collections
import random
import numpy as np
from constant import *

from data_handler import DataFactory

class CrossDataFactory(object):
    def __init__(self, s_path, t_path):
        self.s_path = s_path
        self.t_path = t_path
        self.dictionary = dict()
        self.s_data = []
        self.s_labels = []
        self.t_data = []
        self.t_labels = []
        self.s_input = []
        self.t_input = []
        self.s_input_length = []
        self.t_input_length = []
        self.s_batch_id = 0
        self.t_batch_id = 0
        self.table_size = 100000000
        self.table = np.zeros(self.table_size, dtype=np.int)

    def load_data(self,):
        s_text_data = []
        t_text_data = []
        self.count = [['UNK', -1]]

        with open(self.s_path) as f:
            contents = f.readlines()
            print 'Total {0} reviews in {1}'.format(len(contents), self.s_path)
            d = dict()
            print 'loading training data...'
            for i, line in enumerate(contents):
                if i % 1000 == 0:
                    print i
                t_list = line.strip().split('\t')
                if len(t_list) != 2:
                    print line
                    continue

                text, label = t_list[0], t_list[1]

                words = text.split()
                for word in words:
                    if word in d:
                        d[word] += 1
                    else:
                        d[word] = 1
                s_text_data.append(words)
                self.s_labels.append(float(label))


        with open(self.t_path) as f:
            contents = f.readlines()
            print 'Total {0} reviews in {1}'.format(len(contents), self.t_path)
            print 'loading training data...'
            for i, line in enumerate(contents):
                if i % 1000 == 0:
                    print i
                t_list = line.strip().split('\t')
                if len(t_list) != 2:
                    print line
                    continue

                text, label = t_list[0], t_list[1]

                words = text.split()
                for word in words:
                    if word in d:
                        d[word] += 1
                    else:
                        d[word] = 1
                t_text_data.append(words)
                self.t_labels.append(float(label))
        counter = collections.Counter(d)
        self.count.extend(counter.most_common(vocabulary_size - 1))

        for word, _ in self.count:
            self.dictionary[word] = len(self.dictionary)

        unk_count = 0
        for words in s_text_data:
            t_list = []
            for word in words:
                index = self.dictionary.get(word, 0)
                if index == 0:
                    unk_count += 1
                t_list.append(index)
            self.s_data.append(t_list)
            l = min(len(t_list), max_length)
            self.s_input_length.append(l)
            if len(t_list) >= max_length:
                self.s_input.append(t_list[:max_length])
            else:
                t_list += [0] * (max_length - len(t_list))
                self.s_input.append(t_list)
            
        self.s_words_list = []
        self.s_y_list = []
        self.s_left_list = []
        self.s_right_list = []

        for i in xrange(len(self.s_input_length)):
            for j in xrange(self.s_input_length[i]):
                self.s_words_list.append(self.s_input[i][j])
                self.s_y_list.append(self.s_labels[i])
                if j == 0:
                    self.s_left_list.append([0])
                else:
                    self.s_left_list.append([self.s_input[i][j - 1]])
                    
                if j == self.s_input_length[i] - 1:
                    self.s_right_list.append([0])
                else:
                    self.s_right_list.append([self.s_input[i][j + 1]])

        for words in t_text_data:
            t_list = []
            for word in words:
                index = self.dictionary.get(word, 0)
                if index == 0:
                    unk_count += 1
                t_list.append(index)
            self.t_data.append(t_list)
            l = min(len(t_list), max_length)
            self.t_input_length.append(l)
            if len(t_list) >= max_length:
                self.t_input.append(t_list[:max_length])
            else:
                t_list += [0] * (max_length - len(t_list))
                self.t_input.append(t_list)

        self.t_words_list = []
        self.t_y_list = []
        self.t_left_list = []
        self.t_right_list = []

        for i in xrange(len(self.t_input_length)):
            for j in xrange(self.t_input_length[i]):
                self.t_words_list.append(self.t_input[i][j])
                self.t_y_list.append(self.t_labels[i])
                if j == 0:
                    self.t_left_list.append([0])
                else:
                    self.t_left_list.append([self.t_input[i][j - 1]])
                    
                if j == self.t_input_length[i] - 1:
                    self.t_right_list.append([0])
                else:
                    self.t_right_list.append([self.t_input[i][j + 1]])
        
        self.count[0][1] = unk_count

        self.pow_count = np.zeros(vocabulary_size, dtype=np.int32)

        count_index = 0
        for word, c in self.count:
            self.pow_count[count_index] = c
            count_index += 1

        self.pow_count = np.power(self.pow_count, 0.75)
        self.pow_count_distribution = self.pow_count / np.sum(self.pow_count)
        
        word_index = 0
        t_sum = self.pow_count_distribution[word_index]
        for i in xrange(int(self.table_size)):
            # print i, word_index, t_sum
            self.table[i] = word_index
            if i / float(self.table_size) > t_sum:
                word_index += 1
                t_sum += self.pow_count_distribution[word_index]

            if word_index >= vocabulary_size:
                word_index = vocabulary_size - 1

        self.reversed_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))
        self.shuffle()

    def get_next_batch(self,):
        s_batch_words, s_batch_y_labels, s_batch_left_labels, s_batch_right_labels, s_batch_left_negative_samples, s_batch_right_negative_samples = self.get_next_batch_s()
        t_batch_words, t_batch_y_labels, t_batch_left_labels, t_batch_right_labels, t_batch_left_negative_samples, t_batch_right_negative_samples = self.get_next_batch_t()
        return s_batch_words, s_batch_y_labels, s_batch_left_labels, s_batch_right_labels, s_batch_left_negative_samples, s_batch_right_negative_samples, t_batch_words, t_batch_y_labels, t_batch_left_labels, t_batch_right_labels, t_batch_left_negative_samples, t_batch_right_negative_samples

    def get_next_batch_s(self,):
        if self.s_batch_id == len(self.s_words_list):
            self.s_batch_id = 0
        end_index = min(self.s_batch_id + batch_size, len(self.s_words_list))
        
        batch_words = self.s_words_list[self.s_batch_id : end_index]
        batch_y = self.s_y_list[self.s_batch_id : end_index]
        batch_left = self.s_left_list[self.s_batch_id : end_index]
        batch_right = self.s_right_list[self.s_batch_id : end_index]
        batch_left_negative_samples = np.zeros((len(batch_words), num_sampled), dtype=np.int32)
        batch_right_negative_samples = np.zeros((len(batch_words), num_sampled), dtype=np.int32)

        for i in xrange(len(batch_words)):
            j = 0
            while j < num_sampled:
                random_word = self.sample_word()
                if random_word == batch_left[i]:
                    continue
                else:
                    batch_left_negative_samples[i][j] = random_word
                    j += 1

        for i in xrange(len(batch_words)):
            j = 0
            while j < num_sampled:
                random_word = self.sample_word()
                if random_word == batch_right[i]:
                    continue
                else:
                    batch_right_negative_samples[i][j] = random_word
                    j += 1


        assert num_skips <= 2 * skip_window
        span = 2 * skip_window + 1

        self.s_batch_id = min(self.s_batch_id + batch_size, len(self.s_words_list))
        return batch_words, batch_y, batch_left, batch_right, batch_left_negative_samples, batch_right_negative_samples

    def get_next_batch_t(self,):
        if self.t_batch_id == len(self.t_words_list):
            self.t_batch_id = 0
        end_index = min(self.t_batch_id + batch_size, len(self.t_words_list))
        
        batch_words = self.t_words_list[self.t_batch_id : end_index]
        batch_y = self.t_y_list[self.t_batch_id : end_index]
        batch_left = self.t_left_list[self.t_batch_id : end_index]
        batch_right = self.t_right_list[self.t_batch_id : end_index]
        batch_left_negative_samples = np.zeros((len(batch_words), num_sampled), dtype=np.int32)
        batch_right_negative_samples = np.zeros((len(batch_words), num_sampled), dtype=np.int32)

        for i in xrange(len(batch_words)):
            j = 0
            while j < num_sampled:
                random_word = self.sample_word()
                if random_word == batch_left[i]:
                    continue
                else:
                    batch_left_negative_samples[i][j] = random_word
                    j += 1

        for i in xrange(len(batch_words)):
            j = 0
            while j < num_sampled:
                random_word = self.sample_word()
                if random_word == batch_right[i]:
                    continue
                else:
                    batch_right_negative_samples[i][j] = random_word
                    j += 1
        
        assert num_skips <= 2 * skip_window
        span = 2 * skip_window + 1

        self.t_batch_id = min(self.t_batch_id + batch_size, len(self.t_words_list))
        return batch_words, batch_y, batch_left, batch_right, batch_left_negative_samples, batch_right_negative_samples

    def sample_word(self,):
        random_num = random.randint(0, self.table_size - 1)
        return self.table[random_num]
    
    def shuffle(self,):
        self.s_words_list = np.array(self.s_words_list, np.int32)
        self.s_y_list = np.array(self.s_y_list, np.int32)
        self.s_left_list = np.array(self.s_left_list, np.int32)
        self.s_right_list = np.array(self.s_right_list, np.int32)
        index_shuffle = np.arange(len(self.s_words_list))
        np.random.shuffle(index_shuffle)
        self.s_words_list = self.s_words_list[index_shuffle]
        self.s_y_list = self.s_y_list[index_shuffle]
        self.s_left_list = self.s_left_list[index_shuffle]
        self.s_right_list = self.s_right_list[index_shuffle]

        self.t_words_list = np.array(self.t_words_list, np.int32)
        self.t_y_list = np.array(self.t_y_list, np.int32)
        self.t_left_list = np.array(self.t_left_list, np.int32)
        self.t_right_list = np.array(self.t_right_list, np.int32)
        index_shuffle = np.arange(len(self.t_words_list))
        np.random.shuffle(index_shuffle)
        self.t_words_list = self.t_words_list[index_shuffle]
        self.t_y_list = self.t_y_list[index_shuffle]
        self.t_left_list = self.t_left_list[index_shuffle]
        self.t_right_list = self.t_right_list[index_shuffle]
