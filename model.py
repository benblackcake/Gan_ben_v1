import tensorflow as tf



class Generator:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def model(self,x):
        #hidden_1
        x = tf.matmul(x, self.weights['gen_hidden1'])
        x = tf.add(x, self.biases['gen_hidden1'])
        x = tf.nn.relu(x)
        x =tf.layers.batch_normalization(x)
        #hidden_2
        # x = tf.matmul(x, self.weights['gen_hidden2'])
        # x = tf.add(x, self.biases['gen_hidden2'])
        # x = tf.nn.relu(x)
        # x = tf.layers.batch_normalization(x)
        #hidden_2
        x = tf.matmul(x, self.weights['gen_hidden2'])
        x = tf.add(x, self.biases['gen_hidden2'])
        x = tf.nn.relu(x)
        x = tf.layers.batch_normalization(x)
        #
        x = tf.reshape(x,[-1,3,3,16])
        x = self.__conv2d__(x, self.weights['conv_hidden_64'], self.biases['conv_hidden_64'])
        x = tf.reshape(x, [-1, 3*3*32])
        #
        #output_layer
        out = tf.matmul(x, self.weights['gen_out'])
        out = tf.add(out, self.biases['gen_out'])
        out = tf.nn.relu(out)
        ##
        return out

    def lossFun(self, discimitor_fake):
        return -tf.reduce_mean(tf.log(discimitor_fake))


    def __conv2d__(self,x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def __maxpool2d__(x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')


class Discriminator:

    def __init__(self,weights, biases):
        self.weights = weights
        self.biases = biases


    def model(self,x):
        #hidden_1
        x = tf.matmul(x, self.weights['disc_hidden1'])
        x = tf.add(x, self.biases['disc_hidden1'])
        x = tf.nn.relu(x)
        #hidden_2
        x = tf.matmul(x, self.weights['disc_hidden2'])
        x = tf.add(x, self.biases['disc_hidden2'])
        x = tf.nn.relu(x)
        x = tf.layers.batch_normalization(x)
        #convolution
        # x = tf.reshape(x,[-1,3,3,16])
        # x = self.__conv2d__(x, self.weights['conv_D_hidden_64'], self.biases['conv_D_hidden_64'])
        # x = tf.reshape(x, [-1, 3*3*32])
        #output
        out = tf.matmul(x, self.weights['disc_out'])
        out = tf.add(out, self.biases['disc_out'])
        out = tf.nn.relu(out)
        #


        return out
    def __conv2d__(self,x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)
    def loss(self,discimitor_real,discimitor_fake):
        return -tf.reduce_mean(tf.log(discimitor_real) + tf.log(1. - discimitor_fake))