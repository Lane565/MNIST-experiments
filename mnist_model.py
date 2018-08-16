import tensorflow as tf
from ops import dense, variable_summaries
from networks import *


class Model:
    def __init__(self, sess, n_classes, img_height, img_width, lr_init,
                 lr_decay, lr_step, lr_min, model_dir, mode):
        self.n_classes = n_classes
        self.model_dir = model_dir

        self.lr_init = lr_init
        self.lr_decay = lr_decay
        self.lr_step = lr_step
        self.lr_min = lr_min

        # Input params
        self.input = tf.placeholder('float32', [None, img_height, img_width, 1])  # [batch_size, ...]
        self.labels = tf.placeholder('float32', [None, n_classes])
        self.keep_prob = tf.placeholder('float32', name='keep_prob')
        self.global_step = tf.Variable(0, trainable=False)

        if mode == 'inception':
            self.net = inception_cnn(self.input, self.keep_prob)
        elif mode == 'resnet':
            self.net = resnet_cnn(self.input, self.keep_prob)
        else:
            self.net = vgg_cnn(self.input, self.keep_prob)

        self.flat = tf.contrib.layers.flatten(self.net)

        #self.dense, self.dense_w, self.dense_b = dense(self.flat, tf.nn.relu, 100)

        self.logits, self.out_w, self.out_b = dense(self.flat, None, self.n_classes, name='out')

        # Define loss
        self.losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.labels)

        # compute mean loss
        self.loss = tf.reduce_mean(self.losses)

        # run optimization
        self.learning_rate_op = tf.maximum(self.lr_min,
                                           tf.train.exponential_decay(
                                               self.lr_init,
                                               self.global_step,
                                               self.lr_step,
                                               self.lr_decay,
                                               staircase=True))

        self.train_op = tf.train.AdamOptimizer(self.learning_rate_op).minimize(self.loss)

        # Evaluate model
        self.predictions = tf.map_fn(tf.nn.softmax, self.logits)
        correct_pred = tf.equal(tf.argmax(self.predictions, axis=1), tf.argmax(self.labels, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self.constructSummary(sess)

    def constructSummary(self, sess):
        variable_summaries(self.loss, 'loss')
        variable_summaries(self.accuracy, 'accuracy')
        variable_summaries(self.learning_rate_op, 'learning_rate')
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.model_dir + 'train', sess.graph)
        self.val_writer = tf.summary.FileWriter(self.model_dir + 'validation', sess.graph)

    def train(self, sess, batch_input, batch_labels, keep_prob, global_step):
        _, loss, accuracy, statistics = sess.run([self.train_op, self.loss, self.accuracy, self.merged],
                                               feed_dict={self.input: batch_input,
                                                          self.labels: batch_labels,
                                                          self.global_step: global_step,
                                                          self.keep_prob: keep_prob})

        self.train_writer.add_summary(statistics, global_step)
        return loss, accuracy

    def predict(self, sess, pred_input):
        output = sess.run(self.predictions, feed_dict={self.input: pred_input,
                                                       self.keep_prob: 1.0})
        return output

    def validate(self, sess, test_input, test_labels, global_step):
        loss, accuracy, statistics = sess.run([self.loss, self.accuracy, self.merged], feed_dict={self.input: test_input,
                                                                         self.labels: test_labels,
                                                                         self.keep_prob: 1.0})
        self.val_writer.add_summary(statistics, global_step)
        return loss, accuracy

    def test(self, sess, test_input, test_labels):
        loss, accuracy, predictions = sess.run([self.loss, self.accuracy, self.predictions],
                                                                    feed_dict={self.input: test_input,
                                                                               self.labels: test_labels,
                                                                               self.keep_prob: 1.0})

        confusion_matrix = tf.confusion_matrix(tf.argmax(predictions, axis=1),
                                               tf.argmax(test_labels, axis=1)).eval(session=sess)

        return loss, accuracy, confusion_matrix
