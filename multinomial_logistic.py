#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
#image tools
from matplotlib import pyplot as plt
from PIL import Image

#file manipulation tools

import os 
from glob import glob

import tensorflow as tf
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)


# In[6]:


graph=tf.Graph()
with graph.as_default():
    #variables
    batch_size=128
    beta=.001     #regularization
    image_size=28
    num_labels=10
    
    #input data
    tf_train_dataset=tf.placeholder(tf.float32, shape=(batch_size,image_size*image_size))
    tf_train_labels=tf.placeholder(tf.float32, shape=(batch_size,num_labels))
    tf_valid_dataset=tf.constant(mnist.validation.images)
    tf_test_dataset=tf.constant(mnist.test.images)
    
    #weight and biases
    w_logit=tf.Variable(tf.truncated_normal([image_size*image_size,num_labels]))
    b_logit=tf.Variable(tf.zeros([num_labels]))
    
    
    def model(data):
        #assembles the NN
        
        return tf.matmul(data,w_logit)+b_logit  #return the output layer
    #train computations
    logits=model(tf_train_dataset)
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=tf_train_labels))
    regularized_loss=tf.nn.l2_loss(w_logit)
    total_loss=loss+beta*regularized_loss
    
    #optimizer
    optimizer=tf.train.GradientDescentOptimizer(0.5).minimize(total_loss)
    
    #predictions
    train_prediction=tf.nn.softmax(logits)
    valid_prediction=tf.nn.softmax(model(tf_valid_dataset))
    test_prediction=tf.nn.softmax(model(tf_test_dataset))
    
    
    
    
    


# In[10]:


def accuracy(predictions,labels):
    return(100.0*np.sum(np.argmax(predictions,1)==np.argmax(labels,1))/predictions.shape[0])
    


# In[11]:


num_steps=5001
with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("initialized")
    for step in range (num_steps):
        
        #generate a minibatch
        batch_data,batch_labels=mnist.train.next_batch(batch_size)
        feed_dict={tf_train_dataset:batch_data,tf_train_labels:batch_labels}
        
        _, l,predictions= session.run([optimizer,loss,train_prediction], feed_dict=feed_dict)
        
        if(step%500==0):
            print("Minibatch loss of step %d:%f" %(step,l))
            print("Minibatch accuracy : %.1f%%" %accuracy(predictions,batch_labels))
            print("Validation accuracy: %.1f%%" %accuracy(valid_prediction.eval(),mnist.validation.labels))
            
        print("Test accuracy:%.1f%%" %accuracy(test_prediction.eval(),mnist.test.labels))


# In[ ]:





# In[ ]:




