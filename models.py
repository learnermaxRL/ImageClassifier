from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import cv2
import os

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras import layers as KL
from keras.models import Model
from keras import regularizers


from generators import CrackImageDataGenerator


"""
This function implements the base image preprocessing function used when
training ResNet/Inception consisting on shifting the image range to -1..1 and
normalizing the intensities. This process introduces illumination
invariance.

Additionally, we remove the horizontal lines artefact by subtracting the mean
from each row. Thus, bright/dark rows get centered around 0. We assume here
images have similar properties per row except for the horizontal lines.

Using this preprocessing function allows the net to focus on the crack
patterns instead of modeling the cell shape.
"""

def avg_preprocess(img):
    ret = img / 127. - 1.
    ret = (ret - np.mean(ret)) / np.std(ret)
    ret = ret - np.mean(ret, axis=1)[:, np.newaxis]

    return ret


"""
This class implements the base class for all our models. It implements the
fit method and provides a common interface to build the networks and to get
the predictions
"""

class BaseModel(BaseEstimator, ClassifierMixin):
    build = True

    def __init__(self):
        pass

    def fit(self, path):
        # Call the function to build the network
        if self.build:
            self.network = self.build_network()

        self.network.summary()

        # Get the train/validation generators and compute the number of
        # steps per epoch in order to traverse the entire dataset on each
        # iteration.
        train_gen = self.get_generator(path, 'train')
        train_steps = self.get_steps(path, 'train') // 10
        val_gen = self.get_generator(path, 'validation')
        val_steps = self.get_steps(path, 'validation')

        # Build the callbacks for early stopping and to save the best
        # parameters in terms of performance on the validation set
        early = EarlyStopping(patience=20)
        checkpoint = ModelCheckpoint(self.keras_path, verbose=1,
                                     save_best_only=True)

        # If the output directory does not exist, build the directories.
        if not os.path.exists(os.path.dirname(self.keras_path)):
            os.makedirs(os.path.dirname(self.keras_path))

        # Try to load the weights to allow warm-start initialization
        try:
            self.network.load_weights(self.keras_path)
        except:
            print('Error while loading network')

        # Fit the network.
        self.network.fit_generator(train_gen,
                                   steps_per_epoch=train_steps,
                                   validation_data=val_gen,
                                   validation_steps=val_steps,
                                   epochs=self.max_epochs,
                                   verbose=2,
                                   callbacks=[early, checkpoint])

        # Load the weights from the best iteration
        self.network.load_weights(self.keras_path)

        return self
    
    def predict(self, imgs):
        imgs_ = [cv2.resize(i, self.image_size[: 2]) for i in imgs]
        imgs_ = [avg_preprocess(i) for i in imgs_]
        imgs_ = np.asarray(imgs_)

        return self.network.predict(imgs_)

    def build_network(self):
        # This function should be implemented by the child classes.
        return None

    def get_generator(self, path, group):
        # Apply all the transformations on the training set.
        # For validation/test this is undersirable
        rotate180 = True if group == 'train' else False
        contrast_gap = 0.05 if group == 'train' else 0.
        width_shift_range = 0.1 if group == 'train' else 0.

        # Build the generator and link it to the image source directory
        gen = CrackImageDataGenerator(rotate180=rotate180,
                                      contrast_gap=contrast_gap,
                                      samplewise_center=False,
                                      samplewise_std_normalization=False,
                                      horizontal_flip=True,
                                      vertical_flip=True,
                                      width_shift_range=width_shift_range,
                                      fill_mode='reflect',
                                      preprocessing_function=avg_preprocess)

        gen = gen.flow_from_directory(os.path.join(path, group),
                                      color_mode=self.color_mode,
                                      target_size=(200, 200),
                                      batch_size=self.generator_batch_size,
                                      classes=['no_cracks', 'cracks'],
                                      shuffle=True, seed=42)

        return gen

    def get_steps(self, path, group):
        total_images = 0
        subpath = os.path.join(path, group)
        for class_ in list(os.walk(subpath))[0][1]:
            next_subpath = os.path.join(subpath, class_)
            total_images += len(list(os.walk(next_subpath))[0][2])
        return int(np.ceil(float(total_images) / self.generator_batch_size))


    def load_weights(self):
        if not hasattr(self, 'network'):
            self.network = self.build_network()
        self.network.load_weights(self.keras_path)

        return self

"""
This model implements a network that consists on:
    1. A first block with a state-of-the-art neural network with possibility
       of pre-training on ImageNet.
    2. A sequence of dense and dropout layers. Dropout is used in order to
       avoid overfitting.

The network is trained in two stages:
    1. We fix the entire base network and train the final dense layers.
    2. We train the entire network end-to-end.

This prevents the optimization process to propagate errors from poorly
initialized dense layers in the final part of the network.
"""

class PretrainedModel(BaseModel):
    def __init__(self, base_model='inception', dense_num=1, dense_width=512,
                 dense_activation='relu', dropout=0.3, l2=1e-2,
                 warm_start='imagenet',
                 max_epochs=500, generator_batch_size=16,
                 keras_path=os.path.join('output', 'models', 'inception.h5')
                 ):

        self.base_model = base_model
        self.dense_num = dense_num
        self.dense_width = dense_width
        self.dense_activation = dense_activation

        self.dropout = dropout
        self.l2 = l2

        self.warm_start = warm_start
        self.max_epochs = max_epochs
        self.generator_batch_size = generator_batch_size
        self.keras_path = keras_path
        self.color_mode = 'rgb'

        # We set the image size to be 200x200. It can be changed but the
        # learning process will be slower and it may not fit on standard
        # GPUs.
        self.image_size = (200, 200, 3)
        self.base_trainable = True
        self.build = True

    def build_network(self):
        # Input layer to receive the images
        input_layer = KL.Input(self.image_size)

        # Base model (Resnet or Inception)
        if self.base_model == 'resnet':
            base = ResNet50(weights=self.warm_start, include_top=False)
        elif self.base_model == 'inception':
            base = InceptionV3(weights=self.warm_start, include_top=False)
        
        # Set the base network as trainable or not depending on the
        # base_trainable parameter
        if self.base_trainable:
            base.trainable = True
        else:
            base.trainable = False

        # Get the network graph from the input layer to the final Pooling
        last_ = base(input_layer)
        last_ = KL.GlobalAveragePooling2D()(last_)

        # Append the Dense and Dropout layers to the network
        for d in range(self.dense_num):
            next_dense = KL.Dense(self.dense_width,
                                  activation=self.dense_activation,
                                  kernel_regularizer=regularizers.l2(self.l2))
            last_ = next_dense(last_)

            if self.dropout != 0:
                last_ = KL.Dropout(self.dropout)(last_)

        # Get the output layer with 2 classes (non-crack, crack)
        output = KL.Dense(2, activation='softmax')(last_)

        # Compile the model. We use the adadelta optimizer since it does not
        # require to fine-tune the learning rate and allows an easy extension
        # of the training set.
        model = Model(inputs=[input_layer], outputs=[output])
        model.compile('adadelta', 'categorical_crossentropy',
                      metrics=['accuracy'], sample_weight_mode=None)

        return model

    def fit(self, path):
        epochs = self.max_epochs

        # Fit the final dense layers on 20% of the iterations with the
        # base network fixed.
        self.max_epochs = int(0.2 * epochs)
        self.build = True
        self.base_trainable = False
        super(PretrainedModel, self).fit(path)

        # Fit the entire network end-to-end on the remaining 80% of the
        # iterations
        self.max_epochs = int(0.8 * epochs)
        self.build = False
        self.base_trainable = True
        super(PretrainedModel, self).fit(path)

        return self


"""
Tradition Convolutional Neural Network with interleaving convolutional and
max-pooling layers and final interleaving Dense and Dropout layers. This was
implemented as a baseline.
"""

class CNN(BaseModel):
    def __init__(self, conv_num=3, conv_activation='relu', conv_filters=32,
                 dense_num=1, dense_width=64,
                 dense_activation='relu', dropout=0.3, l2=1e-2,
                 max_epochs=100, generator_batch_size=16,
                 keras_path=os.path.join('output', 'models', 'cnn.h5')
                 ):

        self.conv_num = conv_num
        self.conv_activation = conv_activation
        self.conv_filters = conv_filters

        self.dense_num = dense_num
        self.dense_width = dense_width
        self.dense_activation = dense_activation

        self.dropout = dropout
        self.l2 = l2

        self.max_epochs = max_epochs
        self.generator_batch_size = generator_batch_size
        self.keras_path = keras_path
        self.color_mode = 'grayscale'

        self.image_size = (200, 200, 3)
        self.base_trainable = True
        self.build = True

    def build_network(self):
        input_layer = KL.Input(self.image_size)

        last_ = input_layer

        for c in range(self.conv_num):
            nfilters = self.conv_filters  * (2 ** c)

            for _ in range(2):
                # We are using "tall" kernels in the first layer in order to
                # handle the vertical aspect of most cracks
                cfs = (7, 3) if c == 0 else (3, 3)
                last_ = KL.Conv2D(nfilters, cfs,
                                  activation=self.conv_activation)(last_)
            last_ = KL.MaxPooling2D((2, 2))(last_)

        last_ = KL.Flatten()(last_)

        for d in range(self.dense_num):
            next_dense = KL.Dense(self.dense_width,
                                  activation=self.dense_activation,
                                  kernel_regularizer=regularizers.l2(self.l2))
            last_ = next_dense(last_)

            # Include a dropout layer between each pair of dense layers in
            # order to avoid overfitting
            if self.dropout != 0:
                last_ = KL.Dropout(self.dropout)(last_)

        # Output layer
        output = KL.Dense(2, activation='softmax')(last_)

        model = Model(inputs=[input_layer], outputs=[output])
        model.compile('rmsprop', 'categorical_crossentropy',
                      metrics=['accuracy'], sample_weight_mode=None)

        return model
