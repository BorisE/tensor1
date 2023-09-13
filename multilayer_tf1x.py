import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import trange  

print ("*"*50)
print("*** Tensorflow version: " + str(tf.__version__) + " ****")
print ("*"*50)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./datasets/MNIST_data/", one_hot=True)


# Dataset statistics
print ("*"*50)
print('Training image data: {0}'.format(mnist.train.images.shape)) # len = mnist.train.images.shape[0]
print('Validation image data: {0}'.format(mnist.validation.images.shape))
print('Testing image data: {0}'.format(mnist.test.images.shape))

print('\nTest Labels: {0}'.format(mnist.test.labels.shape))
labels = np.arange(10)
num_labels = np.sum(mnist.test.labels, axis=0, dtype=np.int)
print('Label distribution:{0}'.format(list(zip(labels, num_labels))))

# Define input placeholder
x = tf.placeholder(tf.float32, [None, 784])
# Define labels placeholder
y_ = tf.placeholder(tf.float32, [None, 10])

# Define hidden layer 1
NumL1 = 500
W1 = tf.Variable(tf.truncated_normal([784, NumL1], stddev = 0.03))
b1 = tf.Variable(tf.truncated_normal([NumL1], stddev = 0.03))
py1 = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))

# Define hidden layer 2
NumL2 = 100
W2 = tf.Variable(tf.truncated_normal([NumL1, NumL2], stddev = 0.03))
b2 = tf.Variable(tf.truncated_normal([NumL2], stddev = 0.03))
py2 = tf.nn.relu(tf.add(tf.matmul(py1, W2), b2))

# Definde hidden layer on 100 parameters
W3 = tf.Variable(tf.truncated_normal([NumL2, 10], stddev = 0.03))
b3 = tf.Variable(tf.truncated_normal([10], stddev = 0.03))

# Output layer
y = tf.matmul(py2, W3) + b3
# Softmax to probabilities
py = tf.nn.softmax(y)


# Loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(py), reduction_indices=[1]))

# Optimizer
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
#optimizer = tf.train.AdamOptimizer(learning_rate = 0.05).minimize(cross_entropy)   #BGD with Adam efficient version

# Create a session object and initialize all graph variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Train the model
# trange is a tqdm function. It's the same as range, but adds a pretty progress bar
epochs = 10
batch_size = 128
total_batch = int(mnist.train.images.shape[0] / batch_size)

for epoch in range(epochs):
    for _ in trange(total_batch): 
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# Test trained model
correct_prediction = tf.equal(tf.argmax(py, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Test accuracy: {0}'.format(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})))


# Get weights
weights1 = sess.run(W1)
weights2 = sess.run(W2)
weights3 = sess.run(W3)

fig, ax = plt.subplots(1, 10, figsize=(10, 2))

for digit in range(10):
    ax[digit].imshow(weights1[:,digit].reshape(28,28), cmap='gray')


weights2 = sess.run(W2)

fig2, ax2 = plt.subplots(1, 10, figsize=(10, 2))

for digit in range(10):
    ax2[digit].imshow(weights2[:,digit].reshape(28,28), cmap='gray')

plt.show()

# Close session to finish
sess.close()
