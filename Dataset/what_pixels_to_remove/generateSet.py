import os

import PIL.Image
if __name__ == "__main__":
    whatIdontlike = set()
    for file in os.listdir("whatidontlike"):
        image = PIL.Image.open(os.path.join("whatidontlike/", file))
        print("Processing: " + file)
        for i in range(image.width):
            for j in range(image.height):
                print(image.getpixel((i,j)))
                whatIdontlike.add(image.getpixel((i,j)))
    print(whatIdontlike)
