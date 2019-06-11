from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import *
from keras.models import Sequential, load_model
from keras import optimizers
from datetime import datetime
import glob
import os

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

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    'data/train',
    batch_size=32,
    class_mode='categorical',
    target_size=(256, 256),
    interpolation='bicubic'
)

test_gen = test_datagen.flow_from_directory(
    'data/test',
    batch_size=32,
    class_mode='categorical',
    target_size=(256, 256),
    interpolation='bicubic'
)

model = None

def create_model():
    global model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(256, 256, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))

    model.add(Dropout(0.2))
    model.add(Dense(9))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])


def save():
    global model
    assert isinstance(model, Sequential)
    model.save(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+'.h5')


def load_latest():
    global model
    assert isinstance(model, Sequential)
    files = glob.glob('./models/*.h5')
    latest = max(files, key=os.path.getctime())
    model = load_model(os.path.join('./models', latest))


def train():
    global model
    assert isinstance(model, Sequential)
    model.fit_generator(
        train_gen,
        steps_per_epoch=1000,
        epochs=10,
        validation_data=test_gen,
        validation_steps=1000
    )