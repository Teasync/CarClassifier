import glob
import os
import time
from datetime import datetime

import keras.backend as K
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import callbacks
from keras import optimizers
from keras.layers import *
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import *
from sklearn.utils import class_weight

model = None
hist = None

# K.set_floatx('float16')
# K.set_epsilon(1e-4)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))

BATCH_SIZE = 100
VIEWINGS = 60000
STEPS_PER_EPOCH = VIEWINGS // BATCH_SIZE

train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_gen = train_datagen.flow_from_directory(
    'data/train',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    target_size=(128, 128),
    interpolation='bicubic'
)

test_gen = test_datagen.flow_from_directory(
    'data/test',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    target_size=(128, 128),
    interpolation='bicubic'
)

class_weights = class_weight.compute_class_weight(
    'balanced',
    np.unique(train_gen.classes),
    train_gen.classes)


def create_model():
    global model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activity_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), activity_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activity_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), activity_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activity_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), activity_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activity_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), activity_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(2048, activity_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1024, activity_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512, activity_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, activity_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(9))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics=['accuracy'])


def save():
    global model
    assert isinstance(model, Sequential)
    date_name = os.path.join("./models", datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    if hist is None:
        model.save(date_name + '.h5')
    else:
        acc = np.format_float_positional(hist.history['acc'][len(hist.history['acc']) - 1], precision=5)
        val_acc = np.format_float_positional(hist.history['val_acc'][len(hist.history['val_acc']) - 1], precision=5)
        loss = np.format_float_positional(hist.history['loss'][len(hist.history['val_acc']) - 1], precision=5)
        model.save(date_name + '_acc_' + str(acc) + '_val_' + str(val_acc) + '_loss_' + str(loss) + '.h5')


def change_lr(val):
    global model
    assert isinstance(model, Sequential)
    K.set_value(model.optimizer.lr, val)


def load_latest():
    global model
    files = glob.glob('./models/*.h5')
    latest = max(files, key=os.path.getctime)
    model = load_model(latest)
    return model


def train(epochs=1):
    global model
    global hist
    assert isinstance(model, Sequential)

    hist = model.fit_generator(
        train_gen,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=epochs,
        validation_data=test_gen,
        validation_steps=STEPS_PER_EPOCH // 4,
        workers=12,
        class_weight=class_weights
    )


def train_multi(num):
    for i in range(num):
        train(epochs=1)
        save()
        time.sleep(5)


def load_image(name):
    img = Image.open(name)
    img.load()
    img: Image.Image
    img = img.resize((128, 128))
    data = np.asarray(img, dtype="float32")
    data /= 255
    data = np.expand_dims(data, axis=0)
    return data


def predict(image):
    classes = [dir_name for dir_name in os.listdir('./data/train')]
    classes.sort()
    p = model.predict(image)
    p = [round(e, 4) for e in p[0].tolist()]
    d = dict(zip(classes, p))
    print(d)


# load_latest()
create_model()
model.summary()
# train()
# train_multi(5)
