import argparse
import cv2
import os

import models


def get_recursive_files_(path):
    for _, subpaths, subfiles in os.walk(path):
        for subpath in subpaths:
            for ret in get_recursive_files(os.path.join(path, subpath)):
                yield ret

        for f in subfiles:
            next_ = os.path.join(path, f)
            if os.path.isfile(next_):
                yield next_


def get_recursive_files(path):
    return sorted(list(get_recursive_files_(path)))


def get_args():
    hf = argparse.ArgumentDefaultsHelpFormatter

    parser = argparse.ArgumentParser(description="Predict cell images.",
                                     formatter_class=hf)
    parser.add_argument('--images', metavar="I", nargs='?',
                        default=os.path.join('data', 'partitions',
                                             'test'),
                        help='Path to image folder or filename')
    parser.add_argument('--weights', metavar="w", nargs='?',
                        default=os.path.join('output', 'models',
                                             'inception.h5'),
                        help='Path to the weights file')


    return parser.parse_args()


args = get_args()

# Load the model
model = models.PretrainedModel(keras_path=args.weights)
model.load_weights()

# Get the images
filenames = []

if os.path.isfile(args.images):
    filenames = [args.imgages]
elif os.path.isdir(args.images):
    filenames = get_recursive_files(args.images)

filenames = sorted(filenames)

images = [cv2.imread(f) for f in filenames]

# Get the predictions
preds = model.predict(images)

for f, p in zip(filenames, preds):
    print(','.join([f] + list(map(lambda x: '%.4f' % x, p))))
