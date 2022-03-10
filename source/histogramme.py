import PIL.ImageOps
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from PIL import ImageFilter

# Loading the image
img = Image.open("..\\resources\\simba.png")
img2 = PIL.ImageOps.autocontrast(img)
img_mat = np.array(img2)

# Histogram
n, bins, patches = plt.hist(img_mat.flatten(), bins=range(256))
plt.show()

# Showing the modified image
img2.show()

# Gaussian noise
noise = np.random.normal(0, 7, img_mat.shape)   # img_mat.shape => height x width
noisy_img = Image.fromarray(img_mat + noise).convert('L')   # recreating a new image with noise added
noisy_img.show()

# Trying to remove noise using local treatment
# Box-blur : averaging pixel value according to the 8 surrounding pixels values
noisy_img.filter(ImageFilter.BoxBlur(1)).show()
