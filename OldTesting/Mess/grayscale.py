import os
import pickle

from PIL import Image
import numpy
import bz2

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
                image = numpy.array(Image.open(filepath).convert("L")).astype("float32")
                out = (image - 127.5) / 127.5
                db.append(out)
                print(amnt)

    file = open("grey.bz2", "wb")
    pickle.dump(db,file)
    file.flush()
    file.close()


