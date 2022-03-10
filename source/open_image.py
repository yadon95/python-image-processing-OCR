# Yanis Keto
# 10/03/2022

import numpy as np

from PIL import Image

# Loading the image simba.png

img = Image.open("..\\resources\\simba.png")

img.show()  # Displaying the loaded image

# Printing image's size
w, h = img.size
print("Width : {} px, Height : {} px".format(w, h))

# image's quantification mode
# https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes
print("Format des pixels : {}".format(img.mode))
px_value = img.getpixel((20, 100))
print("Pixel's value located at (20, 100) : {}".format(px_value))

mat = np.array(img)

print(mat)
print("Size of the pixel mat : {}".format(mat.shape))
