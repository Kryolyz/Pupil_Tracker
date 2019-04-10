# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 15:55:39 2019

@author: Daniel
"""

import tensorflow as tf
import numpy as np

kernel1 = 5
filtercount = 20
outs = 2
dimx = 60
dimy = 35
epochs = 2000

with tf.device('/gpu:0'):
    def conv2d(x,W,b,strides=1):
        x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1], padding='SAME')
        x = tf.nn.bias_add(x,b)
        return x#tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')


with tf.device('/gpu:0'):
    def net(x, cw, fw, cb, cw2, cb2, fw2):
        conv = conv2d(x,cw,cb)
        fc1 = maxpool2d(conv)
        fc1 = conv2d(fc1,cw2,cb2)
        fc1 = maxpool2d(fc1)
        fc1 = tf.reshape(fc1, [-1,fw.get_shape().as_list()[0]])
        fc1 = tf.matmul(fc1, fw)
        fc1 = tf.nn.relu(fc1)
#        fc1 = tf.matmul(fc1, fw2)
        return fc1

convW = tf.Variable(tf.ones([kernel1,kernel1,3,20]))
fullycW = tf.Variable(tf.ones([9*15*50,2]))
fullycW2 = tf.Variable(tf.ones([512,2]))
convW2 = tf.Variable(tf.ones([kernel1,kernel1,20,50]))
biasc = tf.Variable(tf.zeros(20))
biasc2 = tf.Variable(tf.zeros(50))

x = tf.placeholder(shape=[None, dimy,dimx,3], dtype=tf.float32)
y = tf.placeholder(shape=[None, outs], dtype=tf.float32)
outs = net(x,convW,fullycW,biasc,convW2,biasc2, fullycW2)
loss = tf.losses.mean_squared_error(y,outs)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
update = optimizer.minimize(loss)

inputdata = np.asarray(topright+bottleft+topmid+topleft+midright+midmid+midleft+toprightbet+topleftbet+bottright+bottmid+botrightbet+botleftbet)
labels = np.zeros(shape=[inputdata.shape[0], 2])
start = 0
end = len(topright)
labels[start:end] = 0
start = end
end = end + len(bottleft)
labels[start:end] = [1920,1080]
start = end
end = end + len(topmid)
labels[start:end] = [1920/2,0]
start = end
end = end + len(topleft)
labels[start:end] = [1920,0]
start = end
end = end + len(midright)
labels[start:end] = [0,1080/2]
start = end
end = end + len(midmid)
labels[start:end] = [1920/2,1080/2]
start = end
end = end + len(midleft)
labels[start:end] = [1920,1080/2]
start = end
end = end + len(toprightbet)
labels[start:end] = [1920/4,1080/4]
start = end
end = end + len(topleftbet)
labels[start:end] = [1920*0.75,1080/4]
start = end
end = end + len(bottright)
labels[start:end] = [0,1080]
start = end
end = end + len(bottmid)
labels[start:end] = [1920/2,1080]
start = end
end = end + len(botrightbet)
labels[start:end] = [1920*0.25,1080*0.75]
start = end
end = end + len(botleftbet)
labels[start:end] = [1920*0.75,1080*0.75]

testdata = np.asarray(midmid+topleftbet)
testlabels = np.zeros(shape=[testdata.shape[0],2])
testlabels[0:len(midmid)] = [0.5*1920,0.5*1080]
testlabels[len(midmid):len(midmid+topleftbet)] = [0.25*1920,0.25*1080]

l = []
testloss = []
outputs = np.zeros([testdata.shape[0], 2], dtype = np.float32)

indexArray = np.arange(inputdata.shape[0])
np.random.shuffle(indexArray)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(3):
        if i == 0:    
            feed_dict = {x:inputdata[indexArray[0:2000]],y:labels[indexArray[0:2000]]}
        elif i == 1:
            feed_dict = {x:inputdata[indexArray[2000:4800]],y:labels[indexArray[2000:4800]]}
        elif i == 2:
            feed_dict = {x:inputdata[indexArray[4800:7278]],y:labels[indexArray[4800:7278]]}
        for i in range(epochs):
            _,ls = sess.run([update,loss], feed_dict=feed_dict)
            l.append(ls)
    
    for a in range(testdata.shape[0]):
        feed_dict = {x:[testdata[a]],y:[testlabels[a]]}
        ls,o = sess.run([loss,outs], feed_dict=feed_dict)
        testloss.append(ls)
        outputs[a] = o
    
    
    
    
    
    
    
    
    
