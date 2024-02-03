import os.path
import shutil
import time

import PIL.Image
import multiprocessing

callers = {}

def worker(destfolder, shaderpng, actualpng):
    idontlikeit = {(24, 24, 54), (0, 0, 84), (24, 24, 112), (16, 16, 52), (0, 0, 96), (32, 32, 56), (0, 0, 172), (0, 0, 56), (0, 0, 120), (0, 0, 68), (8, 8, 112), (0, 0, 80), (32, 32, 98), (16, 16, 112), (0, 0, 144), (112, 112, 214), (0, 0, 104), (0, 0, 52), (32, 32, 70), (0, 0, 64), (0, 0, 128), (8, 8, 50), (0, 0, 76), (32, 32, 112), (0, 0, 100), (24, 24, 70), (0, 0, 48), (160, 160, 255), (0, 0, 112), (0, 0, 60), (8, 8, 98), (0, 0, 124), (32, 32, 84), (16, 16, 98), (0, 0, 136), (0, 0, 255)}
    #chunks are 128x128
    #if 20% or more of this chunks are above pixels (i dont want water to be 90% of dataset) remove it.

    safe_threshold = 3277

    try:
        pilImage = PIL.Image.open(shaderpng).convert("RGB")


        for i in range(pilImage.width):
            for j in range(pilImage.height):
                what = pilImage.getpixel((i, j))
                if (what in idontlikeit):
                    safe_threshold -= 1
                    if(safe_threshold == 0):
                        pilImage.close()
                        return
        onlyname = os.path.basename(shaderpng).split(".")[0]
        pilImage.save(os.path.join(destfolder, (onlyname + "shader.jpg")))
        pilImage.close()

        pilImage = PIL.Image.open(actualpng).convert("RGB")
        pilImage.save(os.path.join(destfolder, (onlyname + ".jpg")))
        pilImage.close()
        #shutil.copy(actualpng, os.path.join(destfolder, (os.path.basename(shaderpng) + ".jpg")))
    except Exception as e:
        print(e)
        return

if __name__=="__main__":
    not_accepted = 0
    accepted = 0
    pool = multiprocessing.Pool(32)

    filesProcessed = 0
    for biome in ["world"]:
        destfolder = os.path.join("/mnt/2tb/nowyDataset/", biome)
        if(not os.path.exists(destfolder)):
            os.mkdir(destfolder)

        shadersubdir = os.path.join(biome, "a")
        normalsubdir = os.path.join(biome, "t")
        builder = []

        for pngdir in os.listdir(shadersubdir):
            pngPath = os.path.join(shadersubdir, pngdir)
            for image in os.listdir(pngPath):
                if(image.count("z") != 0):
                    continue
                builder.append((destfolder, os.path.join(shadersubdir,pngdir,image), os.path.join(normalsubdir, os.path.basename(pngPath), image)))
                #print(os.path.join(shadersubdir,pngdir,image))
                #worker(destfolder, os.path.join(shadersubdir,pngdir,image), os.path.join(normalsubdir, os.path.basename(pngPath), image))
        tasks = pool.starmap_async(worker, builder)
        while not tasks.ready():
            time.sleep(1)
            print(tasks._number_left)

