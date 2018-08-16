import tensorflow as tf
import tensorflow.contrib.slim as slim


def vgg_cnn(inputs, keep_prob, is_training=True):
    with slim.arg_scope([slim.conv2d], stride=1,
                        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        trainable=is_training):
        with tf.variable_scope('Convolution', [inputs]):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1',
                                  normalizer_fn=slim.batch_norm,
                                  normalizer_params={'is_training': is_training})
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.dropout(net, keep_prob, scope='Dropout')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2',
                                  normalizer_fn=slim.batch_norm,
                                  normalizer_params={'is_training': is_training})
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.dropout(net, keep_prob, scope='Dropout')
            net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3',
                              normalizer_fn=slim.batch_norm,
                              normalizer_params={'is_training': is_training})
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.dropout(net, keep_prob, scope='Dropout')
            return net


def inception_cnn(inputs, keep_prob, is_training=True):
    with slim.arg_scope([slim.conv2d], trainable=is_training, stride=1, padding='SAME'):
        net = slim.conv2d(inputs, 32, [7, 7], scope='conv1')
        net = slim.max_pool2d(net, [3, 3], scope='pool1')
        net = slim.conv2d(net, 64, [3, 3], scope='conv2')

        net = inception_module(net, 'Inception_1')
        net = slim.dropout(net, keep_prob, scope='Dropout')
        net = inception_module(net, 'Inception_2')
        net = inception_module(net, 'Inception_3')
        net = slim.dropout(net, keep_prob, scope='Dropout')
        net = inception_module(net, 'Inception_4')
        net = inception_module(net, 'Inception_5')
        net = slim.dropout(net, keep_prob, scope='Dropout')
        net = slim.avg_pool2d(net, [2, 2], scope='pool2')
    return net


def inception_module(inputs, module='inception_default'):
    with slim.arg_scope([slim.max_pool2d], stride=1, padding='SAME'):
        inputs = slim.max_pool2d(inputs, [3, 3], scope='pool3')
        with tf.variable_scope(module, [inputs]):
                with tf.variable_scope('IBranch_0'):
                    branch0 = slim.conv2d(inputs, 64, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('IBranch_1'):
                    branch1 = slim.conv2d(inputs, 96, [1, 1], scope='Conv2d_0a_1x1')
                    branch1 = slim.conv2d(branch1, 128, [3, 3], scope='Conv2d_0b_3x3')
                with tf.variable_scope('IBranch_2'):
                    branch2 = slim.conv2d(inputs, 16, [1, 1], scope='Conv2d_0a_1x1')
                    branch2 = slim.conv2d(branch2, 32, [5, 5], scope='Conv2d_0b_3x3')
                with tf.variable_scope('IBranch_3'):
                    branch3 = slim.max_pool2d(inputs, [3, 3], scope='MaxPool_0a_3x3')
                    branch3 = slim.conv2d(branch3, 32, [1, 1], scope='Conv2d_0b_1x1')
    return tf.concat(axis=3, values=[branch0, branch1, branch2, branch3])

# def inception_module(inputs, module='inception_default'):
#     with slim.arg_scope([slim.avg_pool2d, slim.max_pool2d], stride=1, padding='SAME'):
#         with tf.variable_scope(module, [inputs]):
#             with tf.variable_scope('IBranch_0'):
#                 branch0 = slim.conv2d(inputs, 96, [1, 1], scope='Conv2d_0a_1x1')
#             with tf.variable_scope('IBranch_1'):
#                 branch1 = slim.conv2d(inputs, 64, [1, 1], scope='Conv2d_0a_1x1')
#                 branch1 = slim.conv2d(branch1, 96, [3, 3], scope='Conv2d_0b_3x3')
#             with tf.variable_scope('IBranch_2'):
#                 branch2 = slim.conv2d(inputs, 64, [1, 1], scope='Conv2d_0a_1x1')
#                 branch2 = slim.conv2d(branch2, 96, [3, 3], scope='Conv2d_0b_3x3')
#                 branch2 = slim.conv2d(branch2, 96, [3, 3], scope='Conv2d_0c_3x3')
#             with tf.variable_scope('IBranch_3'):
#                 branch3 = slim.avg_pool2d(inputs, [3, 3], scope='AvgPool_0a_3x3')
#                 branch3 = slim.conv2d(branch3, 96, [1, 1], scope='Conv2d_0b_1x1')
#     return tf.concat(axis=3, values=[branch0, branch1, branch2, branch3])
#
#
# def reduction_module(inputs, module='reduction_default'):
#     with tf.variable_scope(module, [inputs]):
#         with tf.variable_scope('RBranch_0'):
#             branch0 = slim.conv2d(inputs, 384, [3, 3], stride=2, padding='VALID',
#                                   scope='Conv2d_1a_3x3')
#         with tf.variable_scope('RBranch_1'):
#             branch1 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
#             branch1 = slim.conv2d(branch1, 224, [3, 3], scope='Conv2d_0b_3x3')
#             branch1 = slim.conv2d(branch1, 256, [3, 3], stride=2, padding='VALID',
#                                   scope='Conv2d_1a_3x3')
#         with tf.variable_scope('RBranch_2'):
#             branch2 = slim.max_pool2d(inputs, [3, 3], stride=2, padding='VALID',
#                                       scope='MaxPool_1a_3x3')
#     return tf.concat(axis=3, values=[branch0, branch1, branch2])


def resnet_cnn(inputs, keep_prob, is_training=True):
    with slim.arg_scope([slim.conv2d], trainable=is_training, stride=1, padding='SAME'):
        net = slim.conv2d(inputs, 64, [7, 7], scope="conv1", normalizer_fn=slim.batch_norm,
                          normalizer_params={'is_training': is_training})
        net = slim.max_pool2d(net, [3, 3], scope='pool1')

        net = resnet_block(net, 64, 64, is_training, block='a')
        net = resnet_block(net, 64, 64, is_training, block='b')
        net = resnet_block(net, 64, 64, is_training, block='c')
        net = slim.dropout(net, keep_prob, scope='Dropout')
        net = resnet_block(net, 64, 128, is_training, block='d')
        net = resnet_block(net, 128, 128, is_training, block='e')
        net = resnet_block(net, 128, 128, is_training, block='f')
        net = slim.dropout(net, keep_prob, scope='Dropout')
        net = resnet_block(net, 128, 256, is_training, block='g')
        net = resnet_block(net, 256, 256, is_training, block='h')
        net = resnet_block(net, 256, 256, is_training, block='i')
        net = slim.dropout(net, keep_prob, scope='Dropout')
        net = slim.avg_pool2d(net, [2, 2], scope='pool2')
    return net


def resnet_block(inputs, num_filters_in, num_filters_out, is_training, block='default', stride=1):
    if num_filters_in != num_filters_out:
        inputs = slim.conv2d(inputs, num_filters_out, [1, 1], scope='conv0' + block, normalizer_fn=slim.batch_norm,
                             normalizer_params={'is_training': is_training})

    with slim.arg_scope([slim.conv2d], trainable=is_training, stride=stride, padding='SAME', scope=block):
        net = slim.conv2d(inputs, num_filters_out, [3, 3], scope=block + "conv1", normalizer_fn=slim.batch_norm,
                          normalizer_params={'is_training': is_training})
        net = slim.conv2d(net, num_filters_out, [3, 3], scope=block + "conv2", normalizer_fn=slim.batch_norm,
                          normalizer_params={'is_training': is_training})

    return tf.add(net, inputs)
