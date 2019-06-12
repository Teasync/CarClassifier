from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import *
from keras.models import Sequential, load_model
from keras import optimizers
from datetime import datetime
from sklearn.utils import class_weight
import keras.backend as K
import glob
import os
import numpy as np

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
    batch_size=60,
    class_mode='categorical',
    target_size=(256, 256),
    interpolation='bicubic'
)

test_gen = test_datagen.flow_from_directory(
    'data/test',
    batch_size=60,
    class_mode='categorical',
    target_size=(256, 256),
    interpolation='bicubic'
)

class_weights = class_weight.compute_class_weight(
           'balanced',
            np.unique(train_gen.classes),
            train_gen.classes)

model = None
hist = None


def create_model():
    global model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(256, 256, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), input_shape=(256, 256, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(2048))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Dropout(0.5))

    model.add(Dense(9))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.00001), metrics=['accuracy'])


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


def train(steps_per_epoch=1000, epochs=1):
    global model
    global hist
    assert isinstance(model, Sequential)
    hist = model.fit_generator(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=test_gen,
        validation_steps=steps_per_epoch/2,
        workers=12,
        class_weight=class_weights
    )


load_latest()
# create_model()
# model.summary()
# train()
