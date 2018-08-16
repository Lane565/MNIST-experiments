import tensorflow as tf
import numpy as np
import itertools
import matplotlib.pyplot as plt


def variable_summaries(var, name):
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)


def dense(inputs, activation, units, initializer=tf.contrib.layers.xavier_initializer(), name='dense'):
    shape = inputs.get_shape().as_list()

    with tf.variable_scope(name):
        w = tf.get_variable('weights', [shape[1], units], tf.float32, initializer)
        b = tf.get_variable('bias', [units], tf.float32, tf.constant_initializer(0.0))

        out = tf.nn.bias_add(tf.matmul(inputs, w), b)

    if activation != None:
        return activation(out), w, b
    else:
        return out, w, b


def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.rcParams.update({'font.size': 16})
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.rcParams.update({'font.size': 18})
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.rcParams.update({'font.size': 16})

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.rcParams.update({'font.size': 12})

