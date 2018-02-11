#-*- coding: utf-8 -*-
# author: shibei00
# email: bshi@se.cuhk.edu.hk

import time
import tensorflow as tf

from cross_data_handler_v2 import CrossDataFactory
from cross_embedding_v2 import CrossEmbeddingModel


if __name__ == '__main__':
    # df = CrossDataFactory('./data/electronics/labelled_reviews.txt', './data/music/labelled_reviews.txt')
    df = CrossDataFactory('./processed/train/books.txt', './processed/train/music.txt')
    df.load_data()

    t0 = time.time()
    with tf.Session() as sess:
        # model = EmbeddingModel()
        model = CrossEmbeddingModel()
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        model.train(sess, df)
    print 'Done train model, cost time: %0.3fs' % (time.time() - t0)
