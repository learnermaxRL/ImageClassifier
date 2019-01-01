import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

path = os.path.join('data', 'partitions', 'train', 'cracks')
imgs = list(os.walk(path))[0][2]
imgs = [cv2.imread(os.path.join(path, i), 0) for i in imgs]
imgs = [cv2.resize(i, (224, 224)) for i in imgs]
imgs = [(i - np.mean(i)) / np.std(i) for i in imgs]

print('Images loaded')

avg_img = np.mean(imgs, axis=0)
#np.save(os.path.join('data', 'avg'), avg_img)

for i in imgs[10: 15]:
    plt.subplot(2, 4, 1)
    plt.imshow(i)
    plt.subplot(2, 4, 2)
    plt.imshow(avg_img)
    plt.subplot(2, 4, 3)
    plt.imshow(i - avg_img)
    plt.subplot(2, 4, 4)
    iaux = i - np.mean(i, axis=1)[:, np.newaxis]
    plt.imshow(iaux)
    
    plt.subplot(2, 4, 5)
    iaux = iaux - np.mean(iaux, axis=0)[np.newaxis, :]
    plt.imshow(iaux)

    plt.show()
