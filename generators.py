"""
This file extends the ImageDataGenerator from Keras in order to allow random
rotations of 180 degrees and constrast stretching using Gamma correction.

The rotations are limited to 0 and 180 degrees since images are horizontally
aligned.
"""

from keras.preprocessing.image import ImageDataGenerator
from skimage.exposure import adjust_gamma
from keras import backend as K
import numpy as np


class CrackImageDataGenerator(ImageDataGenerator):
    def __init__(self,
                 rotate180=True,
                 contrast_gap=0.,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=K.image_data_format()):
        super(CrackImageDataGenerator,
              self).__init__(featurewise_center, samplewise_center,
                             featurewise_std_normalization,
                             samplewise_std_normalization,
                             zca_whitening, zca_epsilon,
                             rotation_range,
                             width_shift_range, height_shift_range,
                             shear_range, zoom_range, channel_shift_range,
                             fill_mode, cval, horizontal_flip,
                             vertical_flip, rescale, preprocessing_function,
                             data_format)
        self.rotate180 = rotate180
        self.contrast_gap = contrast_gap

    def random_transform(self, x, seed=None):
        # Call the parent class' random_transform method to apply all the
        # other transformations
        ret = super(CrackImageDataGenerator, self).random_transform(x, seed)

        if seed is not None:
            np.random.seed(seed)

        # Apply random rotation of 180 degrees
        if self.rotate180 and np.random.random() > 0.5:
            ret = np.rot90(ret, k=2)

        # Apply contrast stretching
        if self.contrast_gap != 0. and np.random.random() > 0.5:
            gap = 0.05
            adjustment = np.random.uniform(1. - gap, 1. + gap)
            ret_before = ret.copy()
            ret = adjust_gamma(ret, gamma=adjustment, gain=1)

        return ret
