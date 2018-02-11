#-*- coding: utf-8 -*-
# author: shibei00
# email: bshi@se.cuhk.edu.hk

import time
import argparse
import tensorflow as tf

from data_generator import DataFactory
from model import EmbeddingModel
from config import Config

def parse_args():
    parser = argparse.ArgumentParser(description='learning domain-sensitive and sentiment-aware word embeddings')
    parser.set_defaults(
        # input and output
        domain1 = None,
        domain2 = None,
        
        # parameter setting
        vocabulary_size = None,
        embedding_dimension = None,
        num_sampled = 5,
        num_skips = 2,
        skip_window = 1,
        batch_size = 32,
        num_steps = 200000,
        display_steps = 1000,
        e_steps = 10000,
        learning_rate = 1.0,
    )

    parser.add_argument('--domain1', type=str, dest='domain1', 
                      help='name of domain1 [None]')
    parser.add_argument('--domain2', type=str, dest='domain2',
                      help='name of domain2 [None]')
    
    parser.add_argument('--vocabulary_size', type=int, dest='vocabulary_size',
                      help='the size of the vocabulary')
    parser.add_argument('--embedding_dimension', type=int, dest='embedding_dimension',
                      help='embedding dimension')
    parser.add_argument('--num_sampled', type=int, dest='num_sampled',
                      help='number of negative samples')
    parser.add_argument('--num_skips', type=int, dest='num_skips',
                      help='number of skip words')
    parser.add_argument('--skip_window', type=int, dest='skip_window',
                      help='size of skip window')
    parser.add_argument('--batch_size', type=int, dest='batch_size',
                      help='size of batch')
    parser.add_argument('--num_steps', type=int, dest='num_steps',
                      help='total iterations')
    parser.add_argument('--display_steps', type=int, dest='display_steps',
                      help='display steps')
    parser.add_argument('--e_steps', type=int, dest='e_steps',
                      help='e steps')
    parser.add_argument('--learning_rate', type=float, dest='learning_rate',
                      help='learning rate')

    c = Config()
    parser.parse_args(namespace=c)
    return c

    
    

if __name__ == '__main__':
    conf = parse_args()
    df = DataFactory(conf)
    df.load_data()
    t0 = time.time()
    with tf.Session() as sess:
        m = EmbeddingModel(conf)
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        m.train(sess, df)
    print 'Done train model, cost time: %0.3fs' % (time.time() - t0)
