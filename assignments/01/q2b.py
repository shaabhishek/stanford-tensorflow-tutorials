""" Starter code for simple logistic regression model for MNIST
with tf.data module
MNIST dataset: yann.lecun.com/exdb/mnist/
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 03
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import time

import utils

# Define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 30
n_train = 60000
n_test = 10000

# Step 1: Read in data
mnist_folder = '../../examples/data/mnist'
# utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=True)

# Step 2: Create datasets and iterator
# create training Dataset and batch it
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(n_train) # if you want to shuffle your data
train_data = train_data.batch(batch_size)

# create testing Dataset and batch it
#############################
########## TO DO ############
#############################
test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.shuffle(n_test)
test_data = test_data.batch(batch_size)

# create iterator for the dataset
iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)

img, label = iterator.get_next()
train_init = iterator.make_initializer(train_data) # initializer for train_data
test_init = iterator.make_initializer(test_data) # initializer for test_data

# 
w1 = tf.get_variable(name="w1", shape=[784,100], initializer=tf.random_normal_initializer(0, 0.01))
b1 = tf.get_variable(name="b1", shape=(1,100), initializer=tf.zeros_initializer())

h = tf.nn.relu(tf.matmul(img, w1) + b1)

w2 = tf.get_variable(name="w2", shape=[100,10], initializer=tf.random_normal_initializer(0, 0.01))
b2 = tf.get_variable(name="b2", shape=(1,10), initializer=tf.zeros_initializer())

logits = tf.matmul(h, w2) + b2

loss = tf.losses.softmax_cross_entropy(label, logits)

predicted_probs = tf.nn.softmax(logits)
correct_predictions = tf.equal(tf.argmax(predicted_probs, 1), tf.argmax(label, 1))
total_correct = tf.reduce_sum(tf.cast(correct_predictions, tf.float32))

optimizer = tf.train.AdamOptimizer().minimize(loss)

# Tensorboard write
writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())


with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    # iterate through the data
    for i in range(n_epochs):
        sess.run(train_init) # drawing samples from train_data
        total_loss = 0
        n_batches = 0

        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print("Epoch: {}, Avg Epoch Loss: {}".format(i+1, l/n_batches))

    sess.run(test_init)
    correct_pred_test = 0
    try:
        while True:
            correct_pred_batch = sess.run(total_correct)
            correct_pred_test += correct_pred_batch
    except tf.errors.OutOfRangeError:
        pass

    print("Accuracy: {0}".format(correct_pred_test/n_test))

writer.close()