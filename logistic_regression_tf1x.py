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
print('Training image data: {0}'.format(mnist.train.images.shape))
print('Validation image data: {0}'.format(mnist.validation.images.shape))
print('Testing image data: {0}'.format(mnist.test.images.shape))
print('28 x 28 = {0}'.format(28*28)) # 28 x 28 = 784

print('\nTest Labels: {0}'.format(mnist.test.labels.shape))
labels = np.arange(10)
num_labels = np.sum(mnist.test.labels, axis=0, dtype=np.int)
print('Label distribution:{0}'.format(list(zip(labels, num_labels))))

# Example image
exampleImage = 1000
print('\nTrain image {0} is labelled one-hot as {1}'.format(exampleImage, mnist.train.labels[exampleImage,:]))
image = np.reshape(mnist.train.images[exampleImage,:],[28,28])
plt.imshow(image, cmap='gray')
plt.show()

# Define input placeholder
x = tf.placeholder(tf.float32, [None, 784])

# Define linear transformation
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b

# Softmax to probabilities
py = tf.nn.softmax(y)

# Define labels placeholder
y_ = tf.placeholder(tf.float32, [None, 10])

# Loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(py), reduction_indices=[1]))

# Optimizer
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

# Create a session object and initialize all graph variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Train the model
# trange is a tqdm function. It's the same as range, but adds a pretty progress bar
for _ in trange(1000): 
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# Test trained model
correct_prediction = tf.equal(tf.argmax(py, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Test accuracy: {0}'.format(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})))


# Get weights
weights = sess.run(W)

fig, ax = plt.subplots(1, 10, figsize=(20, 2))

for digit in range(10):
    ax[digit].imshow(weights[:,digit].reshape(28,28), cmap='gray')
    ax[digit].show()

# Close session to finish
sess.close()
