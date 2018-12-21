""" Starter code for simple linear regression example using placeholders
Created by Chip Huyen (huyenn@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 03
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time

import numpy as np

import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt

import tensorflow as tf

import utils

DATA_FILE = 'data/birth_life_2010.txt'


def huber_loss(labels, predictions, delta=14.0):
    residual = tf.abs(labels - predictions)
    f1 = lambda: 0.5 * tf.square(residual)
    f2 = lambda: delta * residual - 0.5 * delta**2
    loss = tf.cond(tf.less(residual, delta), f1, f2)
    return loss


# Step 1: read in data from the .txt file
data, n_samples = utils.read_birth_life_data(DATA_FILE)
dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))
dataset = dataset.batch(5)

# iterator = dataset.make_one_shot_iterator()
iterator = dataset.make_initializable_iterator()

# Step 2: create placeholders for X (birth rate) and Y (life expectancy)
# Remember both X and Y are scalars with type float
# X, Y = None, None
#############################
########## TO DO ############
#############################
# X = tf.placeholder(tf.float32, name="X")
# Y = tf.placeholder(tf.float32, name="Y")
X, Y = iterator.get_next()

# Step 3: create weight and bias, initialized to 0.0
# Make sure to use tf.get_variable
# w, b = None, None
#############################
########## TO DO ############
#############################
w1 = tf.get_variable("w1" ,initializer=tf.constant(0.0))
w2 = tf.get_variable("w2" ,initializer=tf.constant(0.0))
b = tf.get_variable("b", initializer=tf.constant(0.0))

# Step 4: build model to predict Y
# e.g. how would you derive at Y_predicted given X, w, and b
# Y_predicted = None
#############################
########## TO DO ############
#############################
Y_predicted = w1*X**2 + w2*X + b

# Step 5: use the square error as the loss function
# loss = None
#############################
########## TO DO ############
#############################
# loss = tf.losses.mean_squared_error(Y, Y_predicted)
# loss = huber_loss(Y, Y_predicted)
loss = tf.losses.huber_loss(Y, Y_predicted, delta = 14.0)

# Step 6: using gradient descent with learning rate of 0.001 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

start = time.time()

# Create a filewriter to write the model's graph to TensorBoard
#############################
########## TO DO ############
#############################
writer = tf.summary.FileWriter("./graphs/linreg", tf.get_default_graph())

with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    #############################
    ########## TO DO ############
    #############################
    # sess.run(tf.variables_initializer([w,b]))
    sess.run(tf.global_variables_initializer())

    # Step 8: train the model for 100 epochs
    for i in range(1000):
        sess.run(iterator.initializer)
        total_loss = 0
        n_batches = 0
        try:
            # for x, y in data:
            while True:
                # Execute train_op and get the value of loss.
                # Don't forget to feed in data for placeholders
                # _, loss = ########## TO DO ############
                # _, loss_out = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
                _, loss_out = sess.run([optimizer, loss])
                total_loss += loss_out
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass

        print('Epoch {0}, Loss: {1}'.format(i, total_loss/n_batches))

    # close the writer when you're done using it
    #############################
    ########## TO DO ############
    #############################
    writer.close()
    
    # Step 9: output the values of w and b
    # w_out, b_out = None, None
    #############################
    ########## TO DO ############
    #############################
    w1_out, w2_out, b_out = sess.run([w1, w2, b])
    print("W1: {}, W2: {}, b: {}".format(w1_out, w2_out, b_out))

print('Took: %f seconds' %(time.time() - start))

# uncomment the following lines to see the plot 
plt.plot(data[:,0], data[:,1], 'bo', label='Real data')
# plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='Predicted data')
predicted_data = np.square(data[:,0])* w1_out + data[:,0] * w2_out + b_out
plt.scatter(x=data[:,0], y=predicted_data, c='r', label='Predicted data')
plt.legend()
plt.show()