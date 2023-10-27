import sys
import math
from params import *
import tensorflow as tf
from six.moves import cPickle
from keras.utils import to_categorical
from tensorflow.keras import activations
from utils.misc_utils import RunningStatMeter
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Input
from tensorflow.keras.experimental import CosineDecay
from utils.model_utils import set_random_seed, set_tf_seed
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from PIL import PngImagePlugin
import subprocess

LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 2)


def load_batch(fpath, label_key='labels'):
    with open(fpath, 'rb') as f:
        if sys.version_info < (3,):
            d = cPickle.load(f)
        else:
            d = cPickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def load_cifar10_data():
    import os
    import numpy as np
    from tensorflow.keras.utils import get_file
    import tensorflow.keras.backend as K
    """Loads CIFAR10 dataset.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = P_SEP.join([DATA_DIR, 'cifar-10-batches-py'])

    # Added due to encountering some weirdness on a test machine when preparing CIFAR-10.
    if not os.path.isdir(dirname):
        origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        path = get_file(dirname, origin=origin, extract=True)
        cmd_list = ["tar", "-xzvf", "data/cifar-10-batches-py", "-C", "data/"]
        subprocess.run(cmd_list)
    else:
        path = dirname

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000: i * 10000, :, :, :],
         y_train[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)


def cifar10_train_test(net_maker,
                             epochs=200, init_lr=0.001, batch_size=256,
                             verbose=1, num_runs=3, offset=0,
                             log_f=print, logger=print):

    dev_perf_meter = RunningStatMeter()
    test_perf_meter = RunningStatMeter()

    for ri in range(num_runs):
        log_f("Execute training run {} of {}".format(ri + 1, num_runs))
        set_tf_seed(ri + offset)
        set_random_seed(ri + offset, log_f=log_f)
        tf.keras.backend.clear_session()
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        config.log_device_placement = False
        sess = tf.Session(config=config)
        tf.keras.backend.set_session(sess)
        net = net_maker()

        (train_X, train_y), (test_X, test_y) = load_cifar10_data()
        log_f('Train: X=%s, y=%s' % (train_X.shape, train_y.shape))
        log_f('Test: X=%s, y=%s' % (test_X.shape, test_y.shape))
        train_y = to_categorical(train_y)
        test_y = to_categorical(test_y)

        data_train = ImageDataGenerator(rescale=1. / 255., zoom_range=0.2,
                                        horizontal_flip=True)
        data_test = ImageDataGenerator(rescale=1. / 255.)
        data_train.fit(train_X)
        data_test.fit(train_X)
        train_it = data_train.flow(train_X, train_y,
                                   shuffle=True,
                                   seed=ri + offset, batch_size=batch_size)
        test_it = data_test.flow(test_X, test_y,
                                 shuffle=False,
                                 seed=ri + offset, batch_size=batch_size)

        net.trainable = True

        newInput = Input(shape=(32, 32, 3))

        newOutputs = net(newInput)

        seq_model = Sequential()
        seq_model.add(Activation(activations.softmax))
        newOutputs = seq_model(newOutputs)

        model = Model(newInput, newOutputs)

        model.summary()

        decay_steps = math.ceil(50000 / batch_size) * epochs
        lr_decayed_fn = CosineDecay(init_lr, decay_steps, alpha=0.001)
        opt = RMSprop(learning_rate=lr_decayed_fn, centered=True, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        hist = model.fit_generator(train_it,
                                   epochs=epochs, validation_data=test_it,
                                   shuffle=True, verbose=verbose)

        _, test_acc = model.evaluate_generator(test_it, verbose=verbose)
        dev_acc = max(hist.history['val_acc'])
        logger('run {} dev_acc={}'.format(ri, dev_acc))
        logger('run {} test_acc={}'.format(ri, test_acc))
        dev_perf_meter.update(dev_acc)
        test_perf_meter.update(test_acc)

    return dev_perf_meter, test_perf_meter
