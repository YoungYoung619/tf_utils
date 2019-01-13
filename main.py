#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np

import tfs.ops as tfops

if __name__ == '__main__':
    """Test Function get_top_k"""
    value = tf.random_normal(shape=[30, 30, 30, 3], dtype=tf.float32)  ##random produce some data
    v, p = tfops.get_top_k(value, 2)  ##return the top 2 values and postions

    with tf.Session() as sess:
        v, p = sess.run([v, p])  ##get the top 2 values and postion
        pass

    """Test Function assign"""
    ## bulid the tf graph ##
    # random produce some data #
    raw_tensor = tf.constant(np.random.normal(size=[10, 2, 10, 3]).astype(np.float32))
    top_k_tensor, position = tfops.get_top_k(raw_tensor, k=500)
    #replace the raw_tensor with the new value 1. at the designated position.
    new_tensor = tfops.assign(input=raw_tensor, position=position, value=[1.])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        new_value = sess.run(new_tensor)

    print("ok")
