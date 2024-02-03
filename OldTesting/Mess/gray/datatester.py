import bz2
import pickle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

file = open("grey.bz2", "rb")

train_images = pickle.load(file)
file.close()

fig = plt.figure(figsize=(4, 4))


for i in range(1):
    plt.subplot(4, 4, 0 + 1)
    plt.imshow(train_images[0], cmap='gray')
    plt.subplot(4, 4, 1 + 1)
    plt.imshow((train_images[0] * 255.0).astype(np.uint8))
    plt.subplot(4, 4, 2 + 1)
    plt.imshow(((train_images[0] * 127.5) + 1).astype(np.uint8))
    plt.subplot(4, 4, 3 + 1)
    plt.axis('off')

plt.savefig('ALLCOMPARASION.png')


