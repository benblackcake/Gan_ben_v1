# from __future__ import division, print_function, absolute_import
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model import Generator,Discriminator
import matplotlib.pyplot as plt
import numpy as np

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Params
num_steps = 100000
batch_size = 128
learning_rate = 0.0002

# Network Params
image_dim = 784 # 28*28 pixels
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 100 # Noise data points

def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))
weights = {
    # Generator
    'gen_hidden1': tf.Variable(glorot_init([noise_dim, gen_hidden_dim])),
    'gen_hidden2': tf.Variable(glorot_init([gen_hidden_dim,400])),
    'conv_hidden_64': tf.Variable(glorot_init([5, 5, 16, 32])),
    'gen_out': tf.Variable(glorot_init([800, image_dim])),

    # Discriminator
    'disc_hidden1': tf.Variable(glorot_init([image_dim, disc_hidden_dim])),
    'disc_hidden2': tf.Variable(glorot_init([disc_hidden_dim, 512])),
    # 'conv_D_hidden_64': tf.Variable(tf.random_normal([3, 3, 16, 32])),
    'disc_out': tf.Variable(glorot_init([512, 1])),
}
biases = {
    #Generator
    'gen_hidden1': tf.Variable(tf.zeros([gen_hidden_dim])),
    'gen_hidden2': tf.Variable(tf.zeros([400])),
    'conv_hidden_64': tf.Variable(tf.zeros([32])),
    'gen_out': tf.Variable(tf.zeros([image_dim])),

    #Discriminator
    'disc_hidden1': tf.Variable(tf.zeros([disc_hidden_dim])),
    'disc_hidden2': tf.Variable(glorot_init([512])),
    # 'conv_D_hidden_64': tf.Variable(tf.random_normal([32])),
    'disc_out': tf.Variable(tf.zeros([1])),
}

gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')
disc_input = tf.placeholder(tf.float32, shape=[None, image_dim], name='disc_input')

G = Generator(weights, biases)
D = Discriminator(weights, biases)

G_sample = G.model(gen_input)

D_real = D.model(disc_input)
D_fake = D.model(G_sample)

G_loss = G.lossFun(D_fake)
D_loss = D.loss(D_real, D_fake)

gen_vars = [weights['gen_hidden1'], weights['gen_out'],
            biases['gen_hidden1'], biases['gen_out']]
# Discriminator Network Variables
disc_vars = [weights['disc_hidden1'], weights['disc_out'],
             biases['disc_hidden1'], biases['disc_out']]
# Build Optimizers
optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)
# Create training operations
train_gen = optimizer_gen.minimize(G_loss, var_list=gen_vars)
train_disc = optimizer_disc.minimize(D_loss, var_list=disc_vars)

init = tf.global_variables_initializer()

if __name__=="__main__":
    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)

        for i in range(1, num_steps + 1):
            # Prepare Data
            # Get the next batch of MNIST data (only images are needed, not labels)
            batch_x, _ = mnist.train.next_batch(batch_size)
            # Generate noise to feed to the generator
            z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])

            # Train
            feed_dict = {disc_input: batch_x, gen_input: z}
            _, _, gl, dl = sess.run([train_gen, train_disc, G_loss, D_loss],
                                    feed_dict=feed_dict)
            if i % 1000 == 0 or i == 1:
                print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))

        # Generate images from noise, using the generator network.
        f, a = plt.subplots(4, 10, figsize=(10, 4))
        for i in range(10):
            # Noise input.
            z = np.random.uniform(-1., 1., size=[4, noise_dim])
            g = sess.run([G_sample], feed_dict={gen_input: z})
            g = np.reshape(g, newshape=(4, 28, 28, 1))
            # Reverse colours for better display
            g = -1 * (g - 1)
            for j in range(4):
                # Generate image from noise. Extend to 3 channels for matplot figure.
                img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
                                 newshape=(28, 28, 3))
                a[j][i].imshow(img)

        f.show()
        plt.draw()
        plt.waitforbuttonpress()
