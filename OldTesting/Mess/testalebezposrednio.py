import os
import pickle
import matplotlib.pyplot as plt

from PIL import Image
import numpy
import bz2

def normalize_data(data):
    """
    Normalize data from [0, 255] to [-1, 1].
    """
    return (data / 127.5) - 1

def denormalize_data(data):
    """
    Denormalize data from [-1, 1] to [0, 255].
    """
    return (data + 1) * 127.5

if __name__=="__main__":
    amnt = 0
    db = []
    for dir in os.listdir():
        if(os.path.isfile(dir)):
            continue
        for file in os.listdir(dir):
            filepath = os.path.join(dir,file)
            if(file.count("z") == 2):
                amnt += 1
                image = numpy.array(Image.open(filepath))

                out = normalize_data(image.astype("float32"))

                img = Image.fromarray(numpy.uint8(denormalize_data(out)))
                img.save("test.png")
                plt.subplot(4, 4, 0 + 1)
                plt.imshow(img.convert("RGB"))
                plt.axis('off')
                plt.savefig('AMOGUSPNGXXDXDXD.png')
                print("Should be called once")

                break
        break
