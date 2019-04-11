from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential, Model, load_model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dropout, Flatten, Dense, Conv2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import rmsprop, SGD
import numpy as np
import os

from keras import backend as K
K.set_image_dim_ordering('tf')

exs_model = InceptionV3(include_top=False,
                        weights='imagenet',
                        input_shape=(150, 150,3))

prepare_data = ImageDataGenerator(rescale=1./255)

train_generator = prepare_data.flow_from_directory('dataset/train',
                                                   target_size=(150, 150),
                                                   batch_size=1,
                                                   class_mode=None,
                                                   shuffle=False)

test_generator = prepare_data.flow_from_directory('dataset/test',
                                                  target_size=(150, 150),
                                                  batch_size=1,
                                                  class_mode=None,
                                                  shuffle=False)

if not os.path.exists('prepare_features/'):
    os.makedirs('prepare_features/')
    print('Directory "prepare_features/" has been created\n')

prepare_features_train = exs_model.predict_generator(train_generator,9350)
np.save(open('prepare_features/pr_features_train.npy', 'wb'), prepare_features_train)
prepare_features_test = exs_model.predict_generator(test_generator, 1650)
np.save(open('prepare_features/pr_features_test.npy', 'wb'),prepare_features_test)

train_data = np.load(open('bottleneck_features/bn_features_train.npy', 'rb'))
train_labels = np.array([1] * 4675 + [0] * 4675)
# print(len(train_data))
# print(len(train_labels))
test_data = np.load(open('bottleneck_features/bn_features_test.npy', 'rb'))
test_labels = np.array([1] * 825 + [0] * 825)
# print(len(test_data))
# print(len(test_labels))

fire_detector_model = Sequential()
fire_detector_model.add(Flatten(input_shape=train_data.shape[1:]))
fire_detector_model.add(Dense(512, activation='relu', name='dense_one'))
fire_detector_model.add(Dropout(0.5, name='dropout_one'))
fire_detector_model.add(Dense(256, activation='relu', name='dense_two'))
fire_detector_model.add(Dropout(0.5, name='dropout_two'))
fire_detector_model.add(Dense(128, activation='relu', name='dense_three'))
fire_detector_model.add(Dropout(0.5, name='dropout_three'))
fire_detector_model.add(Dense(64, activation='relu', name='dense_four'))
fire_detector_model.add(Dropout(0.5, name='dropout_four'))
fire_detector_model.add(Dense(1, activation='sigmoid', name='output'))

fire_detector_model.compile(optimizer=rmsprop(lr=1e-4),
                            loss='binary_crossentropy',
                            metrics=['accuracy'])

fire_detector_model.fit(train_data, train_labels,
                        nb_epoch=50, batch_size=32,
                        steps_per_epoch=100,
                        validation_data=(test_data, test_labels))
if not os.path.exists('neural_network/'):
    os.makedirs('neural_network/')

fire_detector_model.save_weights('neural_network/fr_inception_fire_notfire.hdf5')

weights_filename = 'neural_network/fr_inception_fire_notfire.hdf5'

x = Flatten()(exs_model.output)
x = Dense(512, activation='relu', name='dense_one')(x)
x = Dropout(0.5, name='dropout_one')(x)
x = Dense(256, activation = 'relu', name = 'dense_two')(x)
x = Dropout(0.5, name='dropout_two')(x)
x = Dense(128, activation='relu', name='dense_three')(x)
x = Dropout(0.5, name='dropout_three')(x)
x = Dense(64, activation='relu', name='dense_four')(x)
x = Dropout(0.5, name='dropout_four')(x)
new_model=Dense(1, activation='sigmoid', name='output')(x)
model = Model(input=exs_model.input, output=new_model)
model.load_weights(weights_filename, by_name=True)

for layer in exs_model.layers[:205]:
    layer.trainable = False

model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

filepath = "neural_network/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(150, 150),
        batch_size=4,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(
        'dataset/test',
        target_size=(150, 150),
        batch_size=4,
        class_mode='binary')

pred_generator=test_datagen.flow_from_directory('dataset/test',
                                                  target_size=(150,150),
                                                  batch_size=100,
                                                  class_mode='binary')

model.fit_generator(
        train_generator,
        samples_per_epoch=2000,
        nb_epoch=100,
        validation_data=test_generator,
        nb_val_samples=2000,
        callbacks=callbacks_list)

our_model = load_model('neural_network/fire_detector_model.h5')
def get_prediction(img):
    our_model = load_model('neural_network/fire_detector_model.h5')
    test_image = load_img('maxresdefault.jpg', target_size=(150, 150))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = our_model.predict(test_image)
    return result[0]

