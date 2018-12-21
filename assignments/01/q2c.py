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
n_train = 50000
n_test = 1000

notmnist_folder = 'data/'
train, val, test = utils.read_mnist(notmnist_folder, flatten=True, num_train=n_train)

train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.batch(batch_size)

test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.shuffle(n_test)
test_data = test_data.batch(batch_size)

iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
img, label = iterator.get_next() #placeholders for X and Y

# init ops for both test and train iterators
train_init = iterator.make_initializer(train_data)
test_init = iterator.make_initializer(test_data)

# Create model
w1 = tf.get_variable("w1", [784, 100], initializer=tf.random_normal_initializer(0, 0.01))
b1 = tf.get_variable("b1", [1, 100], initializer=tf.zeros_initializer())

w2 = tf.get_variable("w2", [100, 50], initializer=tf.random_normal_initializer(0, 0.01))
b2 = tf.get_variable("b2", [1, 50], initializer=tf.zeros_initializer())

w3 = tf.get_variable("w3", [50, 10], initializer=tf.random_normal_initializer(0, 0.01))
b3 = tf.get_variable("b3", [1, 10], initializer=tf.zeros_initializer())

h1 = tf.nn.relu(tf.matmul(img, w1) + b1)
h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

logits = tf.matmul(h2, w3) + b3

loss = tf.losses.softmax_cross_entropy(label, logits)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

predictions = tf.nn.softmax(logits, 1)
correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(label, 1))
total_correct_predictions = tf.reduce_sum(tf.cast(correct_predictions, tf.float32))

writer = tf.summary.FileWriter("./graphs/notmnist", tf.get_default_graph())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_idx in range(n_epochs):
        sess.run(train_init)
        loss_epoch = 0
        batch_idx = 0
        try:
            while True:
                batch_idx += 1
                _, loss_out = sess.run([optimizer, loss])
                loss_epoch += loss_out
        except tf.errors.OutOfRangeError:
            pass
        
        print("Epoch: {}, Avg Loss:{}".format(epoch_idx, loss_epoch/batch_idx))

    sess.run(test_init)

    accuracy = 0
    try:
        while True:
            total_correct_predictions_out = sess.run(total_correct_predictions)
            accuracy += total_correct_predictions_out
    except tf.errors.OutOfRangeError:
        pass
    print("Accuracy:{}".format(accuracy/n_test))
