"""
This script splits the dataset intro training/validation/test partitions
We assumme the source data is in data/imgs and the target data will be located
in data/partitions.
We follow the stardad 60-20-20 split.
"""

import numpy as np
import shutil
import os


np.random.seed(42)

# Set the source and target path
src_path = os.path.join('data', 'imgs')
dst_path = os.path.join('data', 'partitions')

# Compute the split size
train_size = 0.6
val_size = 0.2
test_size = 1. - train_size - val_size

# For each class, move all the images to the destination file.
# This is done per class in order to keep the data distribution unchanged.
for class_id, class_ in enumerate(['no_cracks', 'cracks']):
    # Get the filenames
    filenames = sorted(list(os.walk(os.path.join(src_path, class_)))[0][2])
    
    # Shuffle the images
    np.random.shuffle(filenames)

    # Split the names into the three sets
    tr_imgs = filenames[: int(len(filenames) * train_size)]
    val_imgs = filenames[int(len(filenames) * train_size):
                         int(len(filenames) * (train_size + val_size))]
    test_imgs = filenames[int(len(filenames) * (train_size + val_size)):]

    for name, group in [('train', tr_imgs), ('validation', val_imgs),
                        ('test', test_imgs)]:
        next_dst_path = os.path.join(dst_path, name, str(class_id))

        try:
            os.makedirs(next_dst_path)
        except:
            pass

        for img in group:
            shutil.move(os.path.join(os.path.join(src_path, class_, img)),
                        os.path.join(os.path.join(next_dst_path, img)))
