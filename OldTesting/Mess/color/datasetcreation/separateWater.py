import os

from PIL import Image, ImageChops
print(os.getcwd())
print(os.listdir())
image = Image.open("datasetcreation/0_10.png")
problem = {(33, 56, 104, 255), (33, 61, 117, 255), (28, 52, 100, 255), (27, 51, 98, 255), (45, 81, 152, 255), (36, 65, 126, 255), (46, 83, 157, 255), (39, 69, 131, 255), (30, 53, 101, 255), (37, 65, 121, 255), (38, 69, 130, 255), (29, 55, 104, 255), (28, 51, 99, 255), (36, 67, 127, 255), (44, 80, 152, 255), (32, 55, 103, 255), (33, 58, 109, 255), (32, 57, 107, 255), (35, 65, 128, 255), (43, 78, 150, 255), (47, 84, 158, 255), (39, 70, 132, 255), (33, 57, 105, 255), (45, 82, 155, 255), (35, 59, 109, 255), (32, 60, 117, 255), (30, 56, 105, 255), (29, 55, 105, 255), (34, 65, 128, 255), (32, 61, 117, 255), (36, 66, 127, 255), (48, 84, 159, 255), (34, 64, 124, 255), (34, 62, 118, 255), (35, 63, 119, 255), (31, 56, 106, 255), (30, 54, 102, 255), (37, 68, 130, 255), (34, 64, 126, 255), (43, 79, 154, 255), (38, 68, 129, 255), (35, 60, 110, 255), (35, 64, 125, 255), (29, 52, 100, 255), (34, 63, 124, 255), (44, 81, 154, 255), (34, 60, 110, 255), (46, 81, 153, 255), (29, 53, 100, 255), (35, 65, 126, 255), (33, 63, 124, 255), (40, 71, 132, 255), (45, 81, 155, 255), (30, 55, 105, 255), (36, 67, 129, 255), (39, 69, 130, 255), (35, 64, 120, 255), (34, 65, 126, 255), (38, 69, 129, 255), (28, 53, 104, 255), (44, 81, 155, 255), (37, 67, 128, 255), (36, 64, 120, 255), (35, 66, 127, 255), (38, 69, 131, 255), (29, 53, 101, 255), (40, 70, 132, 255), (42, 79, 153, 255), (46, 82, 156, 255), (35, 65, 127, 255), (32, 56, 104, 255), (34, 59, 109, 255), (41, 77, 149, 255), (39, 70, 133, 255), (37, 68, 129, 255), (31, 59, 115, 255), (26, 50, 98, 255), (36, 65, 121, 255), (37, 66, 127, 255), (33, 61, 118, 255), (37, 68, 131, 255), (27, 51, 99, 255), (34, 64, 125, 255), (45, 81, 153, 255), (35, 67, 127, 255), (29, 54, 104, 255), (47, 83, 157, 255), (43, 80, 154, 255), (31, 60, 116, 255), (34, 61, 118, 255), (37, 66, 122, 255), (33, 58, 108, 255), (44, 79, 151, 255), (36, 67, 128, 255), (40, 76, 148, 255), (31, 55, 103, 255), (35, 65, 125, 255), (31, 57, 107, 255), (45, 82, 156, 255), (34, 65, 125, 255), (36, 66, 126, 255), (30, 56, 106, 255), (42, 78, 149, 255), (37, 67, 129, 255), (39, 70, 131, 255), (32, 60, 116, 255), (36, 64, 121, 255), (30, 54, 101, 255), (28, 53, 103, 255), (37, 67, 127, 255), (34, 65, 127, 255), (33, 59, 109, 255), (36, 66, 128, 255), (39, 68, 129, 255), (45, 80, 152, 255), (46, 82, 157, 255), (34, 62, 119, 255), (35, 63, 120, 255), (31, 56, 107, 255), (35, 66, 126, 255), (40, 70, 131, 255), (32, 58, 108, 255), (28, 52, 99, 255), (43, 79, 151, 255), (38, 68, 128, 255), (42, 78, 150, 255), (31, 54, 102, 255), (34, 63, 119, 255), (44, 80, 154, 255), (38, 67, 128, 255), (35, 66, 128, 255), (33, 64, 125, 255), (31, 59, 116, 255), (38, 68, 130, 255), (43, 79, 153, 255), (28, 54, 104, 255), (37, 68, 128, 255)}


width, height = image.size
for x in range(width):
    for y in range(height):
        problem.add(image.getpixel((x,y)))

def willberemoved(whatimg:str):
    img = Image.open(whatimg)
    width, height = img.size
    for x in range(width):
        for y in range(height):
            if(img.getpixel((x, y)) in problem):
                print(image.getpixel((x,y)))
                print(problem)
                print(image.mode)

                return True
    return False

for file in os.listdir("datasetcreation"):
    if(file == "wata.png"):
        continue

    if(".py" in file):
        continue
    print(file)
    print(willberemoved("datasetcreation/" + file))