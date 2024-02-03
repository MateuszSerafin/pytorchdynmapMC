import os.path

import PIL.Image

callers = {}


def willberemoved(image):
    for i in range(pilImage.width):
        for j in range(pilImage.height):
            what = pilImage.getpixel((i, j))
            if (what in idontlikeit):
                #if(what not in callers):
                #    callers[what] = 1
                #else:
                #    callers[what] += 1
                return True
    return False

if __name__=="__main__":
    dirs = "badlands bamboojungle birchforest cherrygrove flowerforest mushroomfield oldspruce snowytaiga"

    idontlikeit = {(30, 54, 104), (31, 56, 106), (30, 54, 101), (79, 105, 163), (42, 79, 153), (93, 113, 158), (29, 52, 99), (43, 78, 149), (82, 107, 163), (85, 108, 158), (34, 64, 126), (82, 106, 161), (82, 105, 159), (88, 112, 168), (86, 109, 161), (32, 57, 106), (34, 65, 127), (30, 55, 104), (33, 58, 107), (93, 114, 159), (37, 66, 126), (77, 102, 157), (40, 76, 148), (31, 57, 106), (82, 105, 158), (28, 53, 103), (36, 65, 124), (33, 63, 124), (31, 59, 115), (85, 108, 159), (31, 55, 102), (78, 103, 158), (33, 64, 125), (26, 50, 98)}
    not_accepted = 0
    accepted = 0

    filesProcessed = 0
    for biome in dirs.split(" "):

        destfolder = os.path.join("/mnt/2tb/Projects/TensorFlowDynmap/ProGAN-PyTorch/nowyDataset/", biome)
        os.mkdir(destfolder)


        biomeCounter = 0
        tilessubdir = os.path.join(biome, "t")
        print(tilessubdir)
        for pngdir in os.listdir(tilessubdir):
            pngPath = os.path.join(tilessubdir, pngdir)

            if (biomeCounter > 10000):
                break

            for image in os.listdir(pngPath):
                if (biomeCounter > 10000):
                    break
                if(image.count("z") != 1):
                    continue
                filesProcessed += 1
                try:
                    pilImage = PIL.Image.open(os.path.join(pngPath, image)).convert("RGB")
                except Exception as e:
                    print(str(e))
                    continue
                if(willberemoved(pilImage)):
                    not_accepted += 1
                    pilImage.close()
                    continue
                else:
                    accepted += 1
                pilImage.save(os.path.join(destfolder, (biome + image + ".jpg")))
                pilImage.close()
                biomeCounter += 1
                print(f"Accepted: {accepted}, not accepted: {not_accepted}")

        if(biomeCounter < 10000):
            print(callers)
            raise Exception("Not enough data")

