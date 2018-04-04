import numpy as np
import tensorflow as tf
import data_reader
from matplotlib import pyplot as plt


def conv_layer(input, n_filters, kernel_shape, stride, act=tf.nn.elu):

    hk = [x//2 for x in (kernel_shape if type(kernel_shape) != int else (kernel_shape, kernel_shape))]
    paddings = tf.constant([[0, 0], [hk[0], hk[0]], [hk[1], hk[1]], [0, 0]])
    padd = tf.pad(input, paddings, mode='REFLECT')

    return tf.layers.conv2d(padd, n_filters, kernel_shape, stride,
                            padding='valid', activation=act)


def custom_act(x):

    ret = tf.nn.relu6(x)
    ret /= 6

    return ret


class Segmenter:

    def __init__(self, model_path=None, load=True):

        self.model_path = model_path

        self._gen_graph()
        self.saver = tf.train.Saver()
        self._init_session(load)

    def _init_session(self, load):
        config = tf.ConfigProto(
            device_count={'GPU': 1}
        )
        self.session = tf.Session(config=config)
        if load and self.model_path is not None:
            try:
                self.saver.restore(self.session, self.model_path)
                return
            except:
                pass

        self.session.run(tf.global_variables_initializer())

    def _gen_graph(self):

        # placeholders
        self.input = tf.placeholder(tf.float32, (None, None, None, 3))
        self.target = tf.placeholder(tf.float32, (None, None, None))

        conv1 = conv_layer(self.input, 64, 5, 2)
        conv2 = conv_layer(conv1, 128, 7, 2)
        conv3 = conv_layer(conv2, 128, 13, 2)
        #conv4 = conv_layer(conv3, 128, 15, 1)
        tconv1 = tf.layers.conv2d_transpose(conv3, 128, 13, 2, padding='same', activation=tf.nn.elu)
        tconv2 = tf.layers.conv2d_transpose(tconv1, 64, 7, 2, padding='same', activation=tf.nn.elu)
        tconv3 = tf.layers.conv2d_transpose(tconv2, 32, 5, 2, padding='same', activation=tf.nn.elu)

        conv_out = conv_layer(tconv3, 32, 5, 1, act=tf.nn.elu)
        conv_out = conv_layer(tconv3, 1, 3, 1, act=custom_act)


        self.output = tf.squeeze(conv_out, 3)

        self.loss = tf.reduce_mean(0.5 * tf.square(self.target-self.output), 0)
        #self.loss = tf.reduce_mean(-(self.target*tf.log(self.output+0.001) + (1-self.target)*tf.log(1-self.output-0.001)), 0)


        self.optimizer = tf.train.AdamOptimizer(0.00001).minimize(self.loss)

    def train(self, data, features):
        ret = self.session.run((self.loss, self.optimizer),
                               feed_dict={self.input: data,
                                          self.target: features})
        return ret[0]

    def feed(self, data):
        return self.session.run(self.output, feed_dict={self.input: data})

    def save(self):
        self.saver.save(self.session, self.model_path)

if __name__ == "__main__":

    model_path = "./model/1"

    model = Segmenter(model_path, True)
    data = data_reader.DataStore(500)

    i = 0

    plt.ion()

    while True:

        batch = data.get_batch(2)

        print(np.mean(model.train(batch[0], batch[1])))
        i += 1


        if (i % 50) == 0:

            model.save()

            example = data.get_batch(1)
            img = model.feed(example[0])[0]
            #img[img < .5] = 0
            #img[img >= .5] = 1

            plt.subplot(121)
            plt.imshow(img, vmin=0, vmax=1)
            plt.subplot(122)
            plt.imshow(example[1][0])
            plt.show()
            plt.pause(0.01)

