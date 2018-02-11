#-*- coding:utf-8 -*-
# author: shibei00
# email: bshi@se.cuhk.edu.hk

import math
import datetime
import os
import tensorflow as tf
import numpy as np
from constant import *
from cross_data_handler_v2 import CrossDataFactory

import matplotlib
matplotlib.use('Agg')

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class CrossEmbeddingModel(object):
    def __init__(self):
        with tf.variable_scope('input'):
            self.s_batch_words = tf.placeholder(tf.int32, [None], name='s_words')
            self.s_batch_y_labels = tf.placeholder(tf.int32, [None], name='s_y_labels')
            self.s_left_labels = tf.placeholder(tf.int32, [None, 1], name='s_left_labels')
            self.s_right_labels = tf.placeholder(tf.int32, [None, 1], name='s_right_labels')
            self.s_left_negative_samples = tf.placeholder(tf.int32, [None, num_sampled], name='s_left_negative_samples')
            self.s_right_negative_samples = tf.placeholder(tf.int32, [None, num_sampled], name='s_right_negative_samples')

            self.t_batch_words = tf.placeholder(tf.int32, [None], name = 't_words')
            self.t_batch_y_labels = tf.placeholder(tf.int32, [None], name='t_y_labels')
            self.t_left_labels = tf.placeholder(tf.int32, [None, 1], name='t_left_labels')
            self.t_right_labels = tf.placeholder(tf.int32, [None, 1], name='t_right_labels')
            self.t_left_negative_samples = tf.placeholder(tf.int32, [None, num_sampled], name='t_left_negative_samples')
            self.t_right_negative_samples = tf.placeholder(tf.int32, [None, num_sampled], name='t_right_negative_samples')

        with tf.variable_scope('embedding'):
            self.c_embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_dimension], -1.0, 1.0), name='c_embeddings')
            self.s_embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_dimension], -1.0, 1.0), name='s_emebddings')
            self.t_embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_dimension], -1.0, 1.0), name='t_embeddings')
            self.c_nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_dimension], stddev=1.0 / math.sqrt(embedding_dimension)))
            self.c_nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
            self.s_nce_weights = self.c_nce_weights
            self.s_nce_biases = self.c_nce_biases
            self.t_nce_weights = self.s_nce_weights
            self.t_nce_biases = self.s_nce_biases

            # self.s_nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_dimension], stddev=1.0 / math.sqrt(embedding_dimension)))
            # self.s_nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
            # self.t_nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_dimension], stddev=1.0 / math.sqrt(embedding_dimension)))
            # self.t_nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        #=======validation=========
        self.valid_size = 16
        self.valid_window = 100
        self.valid_examples = np.random.choice(valid_window, valid_size, replace=False)
        self.valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)
        self._build_model()

    def _build_model(self,):
        self.z = tf.placeholder(tf.float32, [vocabulary_size])
        
        self.c_sentiment = tf.Variable(tf.random_uniform([embedding_dimension,], -1.0, 1.0))
        self.c_bias = tf.Variable(0.5)
        # self.s_sentiment = tf.Variable(tf.random_uniform([embedding_dimension,], -1.0, 1.0))
        # self.s_bias = tf.Variable(0.5)
        # self.t_sentiment = tf.Variable(tf.random_uniform([embedding_dimension,], -1.0, 1.0))
        # self.t_bias = tf.Variable(0.5)
        self.s_sentiment = self.c_sentiment
        self.s_bias = self.c_bias
        self.t_sentiment = self.c_sentiment
        self.t_bias = self.c_bias

        self.s_z = tf.nn.embedding_lookup(self.z, self.s_batch_words)
        self.t_z = tf.nn.embedding_lookup(self.z, self.t_batch_words)

        self.s_common_matrix = tf.nn.embedding_lookup(self.c_embeddings, self.s_batch_words)
        self.s_specific_matrix = tf.nn.embedding_lookup(self.s_embeddings, self.s_batch_words)
        self.t_common_matrix = tf.nn.embedding_lookup(self.c_embeddings, self.t_batch_words)
        self.t_specific_matrix = tf.nn.embedding_lookup(self.t_embeddings, self.t_batch_words)
        
        self.s_common_predict = tf.reduce_sum(tf.multiply(self.s_common_matrix, self.c_sentiment), 1) + self.c_bias
        self.s_specific_predict = tf.reduce_sum(tf.multiply(self.s_specific_matrix, self.s_sentiment), 1) + self.s_bias
        self.t_common_predict = tf.reduce_sum(tf.multiply(self.t_common_matrix, self.c_sentiment), 1) + self.c_bias
        self.t_specific_predict = tf.reduce_sum(tf.multiply(self.t_specific_matrix, self.t_sentiment), 1) + self.t_bias

        self.s_left_common_predict = self.predict(self.s_common_matrix, self.s_left_labels, self.c_nce_weights, self.c_nce_biases)
        self.s_right_common_predict = self.predict(self.s_common_matrix, self.s_right_labels, self.c_nce_weights, self.c_nce_biases)
        self.t_left_common_predict = self.predict(self.t_common_matrix, self.t_left_labels, self.c_nce_weights, self.c_nce_biases)
        self.t_right_common_predict = self.predict(self.t_common_matrix, self.t_right_labels, self.c_nce_weights, self.c_nce_biases)

        self.s_left_common_predict_negative = self.predict_negative(self.s_common_matrix, self.s_left_negative_samples, self.c_nce_weights, self.c_nce_biases)
        self.s_right_common_predict_negative = self.predict_negative(self.s_common_matrix, self.s_right_negative_samples, self.c_nce_weights, self.c_nce_biases)
        self.t_left_common_predict_negative = self.predict_negative(self.t_common_matrix, self.t_left_negative_samples, self.c_nce_weights, self.c_nce_biases)
        self.t_right_common_predict_negative = self.predict_negative(self.t_common_matrix, self.t_right_negative_samples, self.c_nce_weights, self.c_nce_biases)

        self.s_left_specific_predict = self.predict(self.s_specific_matrix, self.s_left_labels, self.s_nce_weights, self.s_nce_biases)
        self.s_right_specific_predict = self.predict(self.s_specific_matrix, self.s_right_labels, self.s_nce_weights, self.s_nce_biases)
        self.t_left_specific_predict = self.predict(self.t_specific_matrix, self.t_left_labels, self.t_nce_weights, self.t_nce_biases)
        self.t_right_specific_predict = self.predict(self.t_specific_matrix, self.t_right_labels, self.t_nce_weights, self.t_nce_biases)

        self.s_left_specific_predict_negative = self.predict_negative(self.s_specific_matrix, self.s_left_negative_samples, self.s_nce_weights, self.s_nce_biases)
        self.s_right_specific_predict_negative = self.predict_negative(self.s_specific_matrix, self.s_right_negative_samples, self.s_nce_weights, self.s_nce_biases)
        self.t_left_specific_predict_negative = self.predict_negative(self.t_specific_matrix, self.t_left_negative_samples, self.t_nce_weights, self.t_nce_biases)
        self.t_right_specific_predict_negative = self.predict_negative(self.t_specific_matrix, self.t_right_negative_samples, self.t_nce_weights, self.t_nce_biases)

        self.s_tmp_z = self.s_z * (tf.sigmoid(self.s_common_predict) * tf.cast(self.s_batch_y_labels, tf.float32) + (1.0 - tf.cast(self.s_batch_y_labels, tf.float32)) * (1.0 - tf.sigmoid(self.s_common_predict))) * self.s_left_common_predict * self.s_right_common_predict * self.s_left_common_predict_negative * self.s_right_common_predict_negative
        self.s_tmp_spec = (1.0 - self.s_z) * (tf.sigmoid(self.s_specific_predict) * tf.cast(self.s_batch_y_labels, tf.float32) + (1.0 - tf.cast(self.s_batch_y_labels, tf.float32)) * (1.0 - tf.sigmoid(self.s_specific_predict))) * self.s_left_specific_predict * self.s_right_specific_predict * self.s_left_specific_predict_negative * self.s_right_specific_predict_negative

        self.t_tmp_z = self.t_z * (tf.sigmoid(self.t_common_predict) * tf.cast(self.t_batch_y_labels, tf.float32) + (1.0 - tf.cast(self.t_batch_y_labels, tf.float32)) * (1.0 - tf.sigmoid(self.t_common_predict))) * self.t_left_common_predict * self.t_right_common_predict * self.t_left_common_predict_negative * self.t_right_common_predict_negative
        self.t_tmp_spec = (1.0 - self.t_z) * (tf.sigmoid(self.t_specific_predict) * tf.cast(self.t_batch_y_labels, tf.float32) + (1.0 - tf.cast(self.t_batch_y_labels, tf.float32)) * (1.0 - tf.sigmoid(self.t_specific_predict))) * self.t_left_specific_predict * self.t_right_specific_predict * self.t_left_specific_predict_negative * self.t_right_specific_predict_negative

        self.s_gamma = self.s_tmp_z / (self.s_tmp_z + self.s_tmp_spec + 1e-20)
        self.t_gamma = self.t_tmp_z / (self.t_tmp_z + self.t_tmp_spec + 1e-20)

        self.s_sentiment_common_loss = tf.reduce_mean(tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.s_common_predict, labels=tf.cast(self.s_batch_y_labels, tf.float32)), self.s_gamma))
        self.s_sentiment_specific_loss = tf.reduce_mean(tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.s_specific_predict, labels=tf.cast(self.s_batch_y_labels, tf.float32)), 1.0 - self.s_gamma))

        self.t_sentiment_common_loss = tf.reduce_mean(tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.t_common_predict, labels=tf.cast(self.t_batch_y_labels, tf.float32)), self.t_gamma))
        self.t_sentiment_specific_loss = tf.reduce_mean(tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.t_specific_predict, labels=tf.cast(self.t_batch_y_labels, tf.float32)), 1.0 - self.t_gamma))


        self.s_common_embed_loss = -1 * tf.reduce_mean((tf.log(self.s_left_common_predict + 1e-10) + tf.log(self.s_right_common_predict+ 1e-10) + tf.log(self.s_left_common_predict_negative+ 1e-10) + tf.log(self.s_right_common_predict_negative+ 1e-10)) * self.s_gamma)
        self.s_specific_embed_loss = -1 * tf.reduce_mean((tf.log(self.s_left_specific_predict+ 1e-10) + tf.log(self.s_right_specific_predict+ 1e-10) + tf.log(self.s_left_specific_predict_negative+ 1e-10) + tf.log(self.s_right_specific_predict_negative)+ 1e-10) * (1.0 - self.s_gamma))
        self.t_common_embed_loss = -1 * tf.reduce_mean((tf.log(self.t_left_common_predict+ 1e-10) + tf.log(self.t_right_common_predict+ 1e-10) + tf.log(self.t_left_common_predict_negative+ 1e-10) + tf.log(self.t_right_common_predict_negative+ 1e-10)) * self.t_gamma)
        self.t_specific_embed_loss = -1 * tf.reduce_mean((tf.log(self.t_left_specific_predict+ 1e-10) + tf.log(self.t_right_specific_predict+ 1e-10) + tf.log(self.t_left_specific_predict_negative+ 1e-10) + tf.log(self.t_right_specific_predict_negative+ 1e-10)) * (1.0 - self.t_gamma))

        # self.s_common_embed_loss = -1 * tf.reduce_mean((tf.log(self.s_left_common_predict + 1e-10) + tf.log(self.s_right_common_predict+ 1e-10) + tf.log(self.s_left_common_predict_negative+ 1e-10) + tf.log(self.s_right_common_predict_negative+ 1e-10)))
        # self.s_specific_embed_loss = -1 * tf.reduce_mean((tf.log(self.s_left_specific_predict+ 1e-10) + tf.log(self.s_right_specific_predict+ 1e-10) + tf.log(self.s_left_specific_predict_negative+ 1e-10) + tf.log(self.s_right_specific_predict_negative)+ 1e-10))
        # self.t_common_embed_loss = -1 * tf.reduce_mean((tf.log(self.t_left_common_predict+ 1e-10) + tf.log(self.t_right_common_predict+ 1e-10) + tf.log(self.t_left_common_predict_negative+ 1e-10) + tf.log(self.t_right_common_predict_negative+ 1e-10)))
        # self.t_specific_embed_loss = -1 * tf.reduce_mean((tf.log(self.t_left_specific_predict+ 1e-10) + tf.log(self.t_right_specific_predict+ 1e-10) + tf.log(self.t_left_specific_predict_negative+ 1e-10) + tf.log(self.t_right_specific_predict_negative+ 1e-10)))


        self.loss = self.s_sentiment_common_loss + self.s_sentiment_specific_loss + self.t_sentiment_common_loss + self.t_sentiment_specific_loss + self.s_common_embed_loss + self.s_specific_embed_loss + self.t_common_embed_loss + self.t_specific_embed_loss

        # self.loss = self.s_sentiment_common_loss + self.s_sentiment_specific_loss + self.t_sentiment_common_loss + self.t_sentiment_specific_loss        
        # self.loss = self.s_common_embed_loss + self.s_specific_embed_loss + self.t_common_embed_loss + self.t_specific_embed_loss

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
        # self.check = tf.add_check_numerics_ops()
        self.c_norm = tf.sqrt(tf.reduce_sum(tf.square(self.c_embeddings), 1, keep_dims=True))
        self.s_norm = tf.sqrt(tf.reduce_sum(tf.square(self.s_embeddings), 1, keep_dims=True))
        self.t_norm = tf.sqrt(tf.reduce_sum(tf.square(self.t_embeddings), 1, keep_dims=True))

        self.c_norm_embeddings = self.c_embeddings / self.c_norm
        self.s_norm_embeddings = self.s_embeddings / self.s_norm
        self.t_norm_embeddings = self.t_embeddings / self.t_norm

        # self.check = tf.add_check_numerics_ops()

    def predict(self, batch_embed, batch_labels, nce_weights, nce_biases):
        batch_labels_col = batch_labels[:, 0]
        weights = tf.nn.embedding_lookup(nce_weights, batch_labels_col)
        # biases = tf.nn.embedding_lookup(nce_biases, batch_labels_col)
        # prob = tf.nn.sigmoid(tf.reduce_sum(batch_embed * weights, 1)  + biases)
        prob = tf.nn.sigmoid(tf.reduce_sum(batch_embed * weights, axis=1))
        return prob

    def predict_negative(self, matrix, negative_samples, nce_weights, nce_biases):
        negative_samples_matrix = tf.nn.embedding_lookup(nce_weights, negative_samples)
        negative_prob = tf.transpose(tf.reduce_sum(tf.multiply(tf.transpose(negative_samples_matrix, [1, 0, 2]), matrix), axis=2))
        # return tf.exp(tf.reduce_sum(negative_prob, axis=1) - tf.reduce_sum(tf.log(1.0 + tf.exp(negative_prob)), axis=1))
        return tf.exp(tf.reduce_sum(-1.0 * tf.log(1.0 + tf.exp(negative_prob)), axis=1))
        
    # def embedding_loss(self, weight, embed, left_label, right_label, left_negative_label, right_negative_label, nce_weights, nce_biases):
    #     left_postive_likelihood = 
    #     left_negative_likelihood = 
    #     left_loss = tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=left_label, inputs=embed, num_sampled=num_sampled, num_classes=vocabulary_size)
    #     right_loss = tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=right_label, inputs=embed, num_sampled=num_sampled, num_classes=vocabulary_size)
    #     return (left_loss + right_loss) * weight
    #     # return left_loss + right_loss
        
    def train(self, sess, df):
        print '========================= start training model ========================='
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("log", sess.graph)
        saver = tf.train.Saver()

        # z = np.ones(vocabulary_size) * 0.5
        z = np.random.uniform(0.0, 1.0, vocabulary_size)
        step = 0
        old_loss = []

        average_loss = 0.0
        average_s_sentiment_common_loss = 0.0
        average_s_sentiment_specific_loss = 0.0
        average_t_sentiment_common_loss = 0.0 
        average_t_sentiment_specific_loss = 0.0

        average_s_common_embed_loss = 0.0
        average_s_specific_embed_loss = 0.0
        average_t_common_embed_loss = 0.0
        average_t_specific_embed_loss = 0.0
        
        word_gamma_sum = np.zeros(vocabulary_size)
        word_gamma_count = np.zeros(vocabulary_size)

        while True:
            s_batch_words, s_batch_y_labels, s_batch_left_labels, s_batch_right_labels, s_batch_left_negative_samples, s_batch_right_negative_samples, t_batch_words, t_batch_y_labels, t_batch_left_labels, t_batch_right_labels, t_batch_left_negative_samples, t_batch_right_negative_samples = df.get_next_batch()
            # print s_batch_left_negative_samples, s_batch_right_negative_samples
            # print t_batch_left_negative_samples, t_batch_right_negative_samples
            # for i, word in enumerate(s_batch_words):
            #    print df.reversed_dictionary[s_batch_left_labels[i][0]], df.reversed_dictionary[s_batch_words[i]], df.reversed_dictionary[s_batch_right_labels[i][0]]

            # for i, word in enumerate(t_batch_words):
            #    print df.reversed_dictionary[t_batch_left_labels[i][0]], df.reversed_dictionary[t_batch_words[i]], df.reversed_dictionary[t_batch_right_labels[i][0]]

            feed_dict = {
                self.s_batch_words : s_batch_words,
                self.s_batch_y_labels : s_batch_y_labels,
                self.s_left_labels : s_batch_left_labels,
                self.s_right_labels : s_batch_right_labels,
                self.s_left_negative_samples : s_batch_left_negative_samples,
                self.s_right_negative_samples : s_batch_right_negative_samples,
                self.t_batch_words : t_batch_words,
                self.t_batch_y_labels : t_batch_y_labels,
                self.t_left_labels : t_batch_left_labels,
                self.t_right_labels : t_batch_right_labels,
                self.t_left_negative_samples : t_batch_left_negative_samples,
                self.t_right_negative_samples : t_batch_right_negative_samples,
                self.z : z
            }
            _, loss_val, s_tmp_z, s_tmp_spec, s_gamma, t_gamma = sess.run([self.optimizer, self.loss, self.s_tmp_z, self.s_tmp_spec, self.s_gamma, self.t_gamma], feed_dict=feed_dict)
            left_common_predict, right_common_predict = sess.run([self.s_left_common_predict, self.s_right_common_predict], feed_dict = feed_dict)
            # print 's_tmp_z:'
            # print s_tmp_z
            # print 's_tmp_spec'
            # print s_tmp_spec
            # print left_common_predict
            s_common_matrix = sess.run([self.s_common_matrix,], feed_dict = feed_dict)
            # print left_common_matrix
            # print left_common_predict
            # print right_common_predict
            # print s_common_matrix

            s_sentiment_common_loss, s_sentiment_specific_loss, t_sentiment_common_loss, t_sentiment_specific_loss, s_common_embed_loss, s_specific_embed_loss, t_common_embed_loss, t_specific_embed_loss = sess.run([self.s_sentiment_common_loss, self.s_sentiment_specific_loss, self.t_sentiment_common_loss, self.t_sentiment_specific_loss, self.s_common_embed_loss, self.s_specific_embed_loss, self.t_common_embed_loss, self.t_specific_embed_loss], feed_dict=feed_dict)
            # print s_common_embed_loss, s_specific_embed_loss
            # print step
            # print s_gamma
            # print s_sentiment_common_loss, s_sentiment_specific_loss, t_sentiment_common_loss, t_sentiment_specific_loss, s_common_embed_loss, s_specific_embed_loss, t_common_embed_loss, t_specific_embed_loss

            assert len(s_batch_words) == len(s_gamma)
            assert len(t_batch_words) == len(t_gamma)
            # print t_gamma
            for i, word in enumerate(s_batch_words):
                # print word
                word_gamma_sum[word] += s_gamma[i]
                word_gamma_count[word] += 1

            for i, word in enumerate(t_batch_words):
                word_gamma_sum[word] += t_gamma[i]
                word_gamma_count[word] += 1

            if step % e_steps == 0 and step > 0:
                t_z = word_gamma_sum / (word_gamma_count + 1e-10)
                non_zero_index = np.nonzero(t_z)
                # print t_z[non_zero_index]
                z[non_zero_index] = t_z[non_zero_index]
                # print non_zero_index
                word_gamma_sum = np.zeros(vocabulary_size)
                word_gamma_count = np.zeros(vocabulary_size)

            step += 1
            if step > num_steps:
                print 'training steps exceeds...'
                break
            if old_loss and abs(loss_val - old_loss[-1]) < 0.000001:
                print 'loss converged...'
                # break
            old_loss.append(loss_val)
            
            average_loss += loss_val
            average_s_common_embed_loss += s_common_embed_loss
            average_s_specific_embed_loss += s_specific_embed_loss
            average_t_common_embed_loss += t_common_embed_loss
            average_t_specific_embed_loss += t_specific_embed_loss

            average_s_sentiment_common_loss += s_sentiment_common_loss
            average_s_sentiment_specific_loss += s_sentiment_specific_loss
            average_t_sentiment_common_loss += t_sentiment_common_loss
            average_t_sentiment_specific_loss += t_sentiment_specific_loss

            if step % 1000 == 0:
                if step > 0:
                    average_loss /= 1000
                    average_s_sentiment_common_loss /= 1000
                    average_s_sentiment_specific_loss /= 1000
                    average_t_sentiment_common_loss /= 1000
                    average_t_sentiment_specific_loss /= 1000

                    average_s_common_embed_loss /= 1000
                    average_s_specific_embed_loss /= 1000
                    average_t_common_embed_loss /= 1000
                    average_t_specific_embed_loss /= 1000

                    print 'average loss at step {0} : {1}'.format(step, average_loss)
                    print average_s_sentiment_common_loss, average_s_sentiment_specific_loss, average_t_sentiment_common_loss, average_t_sentiment_specific_loss,average_s_common_embed_loss, average_s_specific_embed_loss, average_t_common_embed_loss, average_t_specific_embed_loss

                    average_loss = 0.0
                    average_s_sentiment_common_loss = 0.0
                    average_s_sentiment_specific_loss = 0.0
                    average_t_sentiment_common_loss = 0.0 
                    average_t_sentiment_specific_loss = 0.0
                    average_s_common_embed_loss = 0.0
                    average_s_specific_embed_loss = 0.0
                    average_t_common_embed_loss = 0.0
                    average_t_specific_embed_loss = 0.0
                    
                    top_k = 20
                    general_words = (-z).argsort()[:top_k]
                    words_list = []
                    for k in xrange(top_k):
                        general_word = df.reversed_dictionary[general_words[k]]
                        words_list.append(general_word + ':' + str(z[general_words[k]]))
                    print 'General words: ' + ' '.join(words_list)
                    
                    index = 0
                    specific_words = z.argsort()
                    words_list = []
                    word_index = 0

                    while word_index < top_k:
                        word_id = specific_words[index]
                        if df.count[word_id][1] > 0:
                            specific_word = df.reversed_dictionary[word_id]
                            words_list.append(specific_word + ':' + str(z[specific_words[index]]))
                            word_index += 1
                        index += 1
                    print 'Specific words: ' + ' '.join(words_list)

        c_embeddings, s_embeddings, t_embeddings = sess.run([self.c_norm_embeddings, self.s_norm_embeddings, self.t_norm_embeddings], feed_dict=feed_dict)

        # my_s_embedding = np.transpose(np.transpose(c_embeddings) * z + np.transpose(s_embeddings) * (1.0 - z))
        # my_t_embedding = np.transpose(np.transpose(c_embeddings) * z + np.transpose(t_embeddings) * (1.0 - z))
        # my_s_embedding = np.concatenate((c_embeddings, s_embeddings), axis = 1)
        # my_t_embedding = np.concatenate((c_embeddings, t_embeddings), axis = 1)
        # my_s_embedding = s_embeddings
        # my_t_embedding = c_embeddings
        my_s_embedding = np.transpose(np.concatenate((np.transpose(c_embeddings) * z, np.transpose(s_embeddings) * (1.0 - z)), axis=0))
        my_t_embedding = np.transpose(np.concatenate((np.transpose(c_embeddings) * z, np.transpose(t_embeddings) * (1.0 - z)), axis=0))

        outpath_directory = './output/'
        now = datetime.datetime.now();
        suffix = now.strftime("%y%m%d-%H%M%S") + "";
        suffix += "/";
        output_directory = os.path.join(outpath_directory, suffix);
        os.makedirs(os.path.abspath(output_directory));
        options_output_file = open(output_directory + "option.txt", 'w');
        domain_str = df.s_path + '\n' + df.t_path
        options_output_file.write(domain_str)
        with open('./constant.py', 'r') as f_config:
            contents = f_config.read()
            options_output_file.write(contents)
        options_output_file.close()

        s_embeddings_file = output_directory + 's_embeddings.txt'
        t_embeddings_file = output_directory + 't_embeddings.txt'
        z_file = output_directory + 'z.txt'

        with open(s_embeddings_file, 'w') as f:
            f.write('{0}\t{1}\n'.format(vocabulary_size, embedding_dimension))
            for i in xrange(vocabulary_size):
                content = df.reversed_dictionary[i] + ' ' + ' '.join([str(x) for x in my_s_embedding[i]]) + '\n'
                f.write(content)
 
        with open(t_embeddings_file, 'w') as f:
            f.write('{0}\t{1}\n'.format(vocabulary_size, embedding_dimension))
            for i in xrange(vocabulary_size):
                content = df.reversed_dictionary[i] + ' ' + ' '.join([str(x) for x in my_t_embedding[i]]) + '\n'
                f.write(content)

        with open(z_file, 'w') as f:
            for i in xrange(vocabulary_size):
                content = df.reversed_dictionary[i] + ' ' + str(z[i]) + '\n'
                f.write(content)

        try:
            # pylint: disable=g-import-not-at-top
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
            plot_only = 500
            low_dim_embs = tsne.fit_transform(s_embeddings[:plot_only, :])
            labels = [df.reversed_dictionary[i] for i in xrange(plot_only)]
            self.plot_with_labels(low_dim_embs, labels, output_directory + 's_embedding.png')

            low_dim_embs = tsne.fit_transform(c_embeddings[:plot_only, :])
            labels = [df.reversed_dictionary[i] for i in xrange(plot_only)]
            self.plot_with_labels(low_dim_embs, labels, output_directory + 'c_embedding.png')

            low_dim_embs = tsne.fit_transform(t_embeddings[:plot_only, :])
            labels = [df.reversed_dictionary[i] for i in xrange(plot_only)]
            self.plot_with_labels(low_dim_embs, labels, output_directory + 't_embedding.png')
        except ImportError as ex:
            print('Please install sklearn, matplotlib, and scipy to show embeddings.')
            print(ex)
            
    # pylint: disable=missing-docstring
    # Function to draw visualization of distance between embeddings.
    def plot_with_labels(self, low_dim_embs, labels, filename):
        assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
        plt.figure(figsize=(18, 18))  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label,
                         xy=(x, y),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')

        plt.savefig(filename)
