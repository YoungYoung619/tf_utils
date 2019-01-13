#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
import nps.ops as npops

def get_top_k(input, k, sorted = True):
    """This function would compare the values of input tensor(any shape) in the lowest
    dimension(axis=-1),and choose top k values and corresponding positions

    Args:
        input：a tensor with any shape
        k：the number of top_k values
        sorted: if true the resulting k elements will be sorted by the values in descending order.
    Return:
        the top k value and corresponding position.

    Example:
        value = tf.random_normal(shape=[30,30,30,3], dtype=tf.float32) ##random produce some data
        v,p = get_top_k(value,2)  ##return the top 2 values and postions

        with tf.Session() as sess:
            v,p = sess.run([v,p]) ##get the top 2 values and postion
    """
    shape = input.get_shape()
    value, pos = tf.nn.top_k(input=tf.reshape(input,[-1]), k=k, sorted=sorted)
    posList = []
    for i in range(len(shape)):
        index = len(shape)-i-1
        posList.append(pos % shape[index])
        if index > 0:
            pos = pos // shape[index]
    posList.reverse()

    posList2 = []
    for i in range(k):
        tempPos = []
        for j in range(len(posList)):
            tempPos.append(posList[j][i])
        posList2.append(tempPos)
    pos = tf.concat(posList2, axis=-1)
    pos = tf.reshape(pos, shape=[k, len(shape)])
    return value,pos


def assign(input, position, value):
    """This function would assign a value to the input in the specific position

    Args:
        input: A tensor with any shape.
        position: Specify the postions of input where you wanna assign the value.
        value: the value you wanna assign in. Must be 1-D array(or tensor,list)
        ---The length of value must be 1 or the same with the position's.
        ---Dtype of value must be the same with input's.

    Return:
        the tensor after modify.

    Example:
        ## bulid the tf graph ##
        # random produce some data #
        raw_tensor = tf.constant(np.random.normal(size=[10,2, 10, 3]).astype(np.float32))
        # get the top 500 values and corresponding positions #
        top_k_tensor, position = get_top_k(raw_tensor, k=500)
        #replace the raw_tensor with the new value 1. at the designated position.
        new_tensor = assign(input=raw_tensor, position=position, value=[1.])

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            new_value = sess.run(new_tensor)
    """
    ## judge the inputs' type##
    if(type(input)!=tf.Tensor):
        raise ValueError('Input value must be a tensor.')

    if type(position)!=list and type(position)!=np.ndarray and type(position)!=tf.Tensor:
        raise ValueError('The type of position must be one of these(tensor, ndarray or list).')

    if type(value)!=list and type(value)!=np.ndarray and type(value)!=tf.Tensor:
        raise ValueError('The type of value must be one of these(tensor, ndarray or list).')

    ## judge whether exist a conflict between position and value ##
    value_lenth = len(value.get_shape()) if type(value)==tf.Tensor else len(value)
    pos_lenth = position.get_shape()[0] if type(position)==tf.Tensor else len(position)
    if value_lenth!=1 and value_lenth!=pos_lenth:
        raise ValueError('The length of value must be 1 or the same with the position\'s.')

    ## judge whether exist a conflict between position and input tensor ##
    pos_dim = position.get_shape()[1] if type(position)==tf.Tensor else np.shape(np.array(position))[-1]
    input_dim = len(input.get_shape())
    if pos_dim!=input_dim:
        raise ValueError('The dimension of input and position must be same.')

    ## convert to ndarray cause the tensor vars counld not be easily modified ##
    inputData = tf.Session().run(input) ##ndarray
    posData = tf.Session().run(position) if type(position)==tf.Tensor else position
    valueDate = tf.Session().run(value) if type(value)==tf.Tensor else value

    ## assign the new values to the input tensor ##
    new_values = npops.assign(input=inputData, position=posData, value=valueDate)

    ## return a tensor after modify ##
    return tf.Variable(new_values)
