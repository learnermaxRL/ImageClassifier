# Detection of Cracks on Solar Cells using Deep Convolutional Neural Networks

## Requirements

We are using Python 3+ for this project. Also, it is recommended to run this
code with available GPUs.
For development, we used a nvidia GeForce GTX 1050 Ti.

The following libraries are used in this project.

| Library                | Version  |
| ---------------------- | --------:|
| h5py                   |    2.7.1 |
| imutils                |    0.4.3 |
| Keras                  |    2.1.1 |
| matplotlib             |    2.1.1 |
| numpy                  |   1.14.0 |
| opencv-contrib-python  | 3.3.0.10 |
| opencv-python          | 3.3.0.10 |
| scikit-image           |   0.13.0 |
| scikit-learn           |   0.19.1 |
| scipy                  |    1.0.0 |
| tensorflow-gpu         |    1.4.0 |
| tensorflow-tensorboard | 0.4.0rc3 |

## Usage

The first step is to build the training/validation/test partitions. In order
to do this, place the images in the folder `data/cells` and execute the command

> python3 build_partitions.py

Then, use the following command to train the model. All the parameters are
optional. If you want to re-train a model with warm-start initialization, use
the `--output` argument to specify the path to the weights to update.
This command will create the model file.

> python3 train.py --img-path data/partitions/ --epochs 100 --output output/models/inception.h5

Then, in order to get the predictions. Use the script `predict.py`. The path to
the images is specified in the argument `--images` where you can specify either
a single image or a folder with a set of images. The output will be a line per
image with the filename and the probabilistic prediction. The model path is
specified using the `--weights` parameter.

> python3 train.py --images data/imgs/ --weights output/models/inception.h5
