from sklearn import metrics
import argparse
import cv2
import os

import models


def get_args():
    hf = argparse.ArgumentDefaultsHelpFormatter

    parser = argparse.ArgumentParser(description="Train the architecture.",
                                     formatter_class=hf)
    parser.add_argument('--img-path', metavar="P", nargs='?',
                        default=os.path.join('data', 'partitions'),
                        help='Path to partitions folder')
    parser.add_argument('--epochs', metavar="E", type=int,
                        default=100,
                        help='Number of epochs to train')
    parser.add_argument('--output', metavar="O", nargs='?',
                        default=os.path.join('output', 'models',
                                             'inception.h5'),
                        help='Path to the output file')

    return parser.parse_args()


args = get_args()

model = models.PretrainedModel(max_epochs=args.epochs, keras_path=args.output)
model.fit(args.img_path)

# Evaluate the performance of the network
print('%15s %9s %9s %9s' % ('subset', 'acc', 'f1', 'roc-auc'))

for subset in ['train', 'validation', 'test']:
    images = []
    labels = []
    for labeld_id, label in enumerate(['no_cracks', 'cracks']):
        subpath = os.path.join(args.img_path, subset, label)
        for f in list(os.walk(subpath))[0][2]:
            path = os.path.join(subpath, f)
            img = cv2.imread(path)
            images.append(img)
            labels.append(labeld_id)
        
    probs = model.predict(images)
    preds = probs[:, 1] > probs[:, 0]

    print('%15s %9.4f %9.4f %9.4f' %
            (subset,
             metrics.accuracy_score(labels, preds),
             metrics.f1_score(labels, preds),
             metrics.roc_auc_score(labels, probs[:, 1]),
             ))
