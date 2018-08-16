import tensorflow as tf
import numpy as np
from mnist_model import Model
import os
import threading
import matplotlib.pyplot as plt
from ops import plot_confusion_matrix
import datetime
import argparse
from tensorflow.examples.tutorials.mnist import input_data
np.set_printoptions(suppress=True)

##CONFIG
# Optimization Parameters
num_epochs = 10
batch_size = 32
dropout_keep_prob = 0.5

lr_init = 0.001
lr_min = 0.00001
lr_n_decrease = 10  # how many times to decrease before lr should reach minimum

img_height = 28
img_width = 28
save_step = 5  # save every ... epochs
display_step = 50  # display/validate every ... steps

# Network Parameters
n_classes = 10  # number of classes
mode = 'vgg'

# General
model_to_load = 'vgg-model-2018-06-25T11_14_11'
timestamp = datetime.datetime.now().isoformat().split('.')[0].replace(':', '_')
model_dir = './experiments/' + mode + '-model-' + timestamp + '/'


def load_mnist():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    X_train = [np.expand_dims(img.reshape([28, 28]), axis=-1) for img in mnist.train.images]
    y_train = mnist.train.labels

    X_val = [np.expand_dims(img.reshape([28, 28]), axis=-1) for img in mnist.validation.images]
    y_val = mnist.validation.labels

    X_test = [np.expand_dims(img.reshape([28, 28]), axis=-1) for img in mnist.test.images]
    y_test = mnist.test.labels

    del mnist

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_batch(X, y, num_batch, size_batch):
    # Get indices
    batch_start_idx = int(num_batch*size_batch)
    batch_end_idx = min(int(num_batch*size_batch + size_batch), len(X))
    # Get batch data
    batch_x = X[batch_start_idx:batch_end_idx]
    batch_y = y[batch_start_idx:batch_end_idx]

    return batch_x, batch_y


def shuffle(X, y):
    zipped_xy = list(zip(X, y))
    np.random.shuffle(zipped_xy)
    return zip(*zipped_xy)


def train(sess, model, X, y, X_val, y_val):
    # Perform training
    print("Start training..")
    saver = tf.train.Saver()

    num_batches = int(len(X) / batch_size)
    if (len(X) % batch_size != 0):
        num_batches += 1

    for epoch in range(num_epochs):
        X, y = shuffle(X, y)
        for batch in range(num_batches):
            batch_x, batch_y = get_batch(X, y, batch, batch_size)

            # Run model on batch
            loss, accuracy = model.train(sess, batch_x, batch_y, dropout_keep_prob,
                                         epoch * num_batches + batch)

            if batch % display_step == 0 or batch == 0:
                val_loss, val_accuracy = model.validate(sess, X_val, y_val, epoch * num_batches + batch)
                print("Epoch " + str(epoch) + ", Batch " + str(batch) + ", Minibatch Loss= " +
                      "{:.6f}".format(loss) + ", Training Accuracy= " +
                      "{:.5f}".format(accuracy) + ", Validation Loss= " +
                      "{:.5f}".format(val_loss) + ", Validation Accuracy= " +
                      "{:.5f}".format(val_accuracy))

        if epoch % save_step == 0 and epoch != 0:
            saver.save(sess, model_dir + 'model-epoch-' + str(epoch) + '.cptk')
            print("Model saved")

    saver.save(sess, model_dir + 'model-epoch-final.cptk')
    print("Training done, final model saved")


def test(sess, model, X, y):
    test_loss, test_accuracy, test_confusion_matrix = model.test(sess, X, y)
    print("Test loss: ", test_loss)
    print("Test Accuracy: ", test_accuracy)
    print("Test Confusion Matrix:")
    print(test_confusion_matrix)

    if not args.load_model:
        # dump test results to model folder
        with open(model_dir + 'evaluation.txt', "w") as file:
            print("Test loss: ", test_loss, file=file)
            print("Test Accuracy: ", test_accuracy, file=file)
            print("Test Confusion Matrix:", file=file)
            print(test_confusion_matrix, file=file)

    plot_confusion_matrix(test_confusion_matrix, classes=np.arange(0, 9))
    plt.show()


def launchTensorBoard():
    import os
    os.system('tensorboard --logdir=' + model_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', help='boolean for loading a model', dest='load_model', type=bool, default=False)
    parser.add_argument('-d', help='directory to load the model', dest='load_dir', type=str,
                        default='C:\\Users\\Fisch003\\PycharmProjects\\capacitive-touchpad-project\\experiments\\')
    args = parser.parse_args()
    load_dir = os.path.join(args.load_dir, model_to_load)
    if args.load_model:
        # start tensorboard visualization of the learning process
        t = threading.Thread(target=launchTensorBoard, args=([]))
        model_dir = load_dir + '/'
        t.start()

    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()

    # determine learning rate schedule
    num_batches = int(len(X_train) / batch_size)
    if (len(X_train) % batch_size != 0):
        num_batches += 1
    lr_decay = np.power((lr_min / lr_init), (1 / lr_n_decrease))
    lr_step = int(num_epochs * num_batches / lr_n_decrease)

    # create model
    sess = tf.Session()
    model = Model(sess, n_classes, img_height, img_width, lr_init,
                  lr_decay, lr_step, lr_min, model_dir, mode)

    # Initialize the variables (i.e. assign their default value)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # output number of parameters for visualizing model complexity
    total_parameters = 0
    print("Number of parameters by variable:")
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = np.prod(shape.as_list())
        print(variable.name + " " + str(shape) + ": " + str(variable_parameters))
        total_parameters += variable_parameters
    print("Total number of model parameters: " + str(total_parameters))

    if not args.load_model:
        # start tensorboard visualization of the learning process
        t = threading.Thread(target=launchTensorBoard, args=([]))
        t.start()

        # train model
        train(sess, model, X_train, y_train, X_val, y_val)
    else:
        # load model
        restore_model = tf.train.Saver()
        try:
            restore_model.restore(sess, os.path.join(load_dir, "model-epoch-final.cptk"))
            print("Model restored.")
        except Exception as e:
            print("Model not restored: ", str(e))
            exit(0)

    # evaluate model
    test(sess, model, X_test, y_test)




