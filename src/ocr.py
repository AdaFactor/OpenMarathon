import os
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from face_detection import cropped_images
from skimage.color import rgb2gray
from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank, gaussian, frangi
from skimage.measure import find_contours
from skimage.feature import canny

# Import Region of interest
cropped = cropped_images(image_path='../data/001.jpg')

# Prepoceesing
for crop in cropped:
        # RGB2GRAY
        gray_image = rgb2gray(crop)
        
        # Global Otsu
        threshold_global_otsu = threshold_otsu(gray_image)
        global_otsu = gray_image >= threshold_global_otsu
        im = gaussian(global_otsu, sigma=2)
        # im = global_otsu
        im = frangi(im)

        # Find Contours
        contours = find_contours(im, 0.0000001)
        contours_lens = np.sort([ len(c) for c in contours ])
        print(contours_lens)
        lower_bound = np.mean(contours_lens)
        upper_bound = contours_lens[-4]
        print('lower:', lower_bound, 'upper:', upper_bound)
        contours = [ c for c in contours if len(c[:, 0]) > lower_bound and len(c[:, 0]) < upper_bound]
        print('Len:', len(contours))

        fig, ax = plt.subplots()
        ax.imshow(im, cmap=plt.cm.gray)
        for n, contour in enumerate(contours):
            x, y = min(contour[:, 1]), min(contour[:, 0])
            show_text = str(n)
            ax.text(x, y, show_text, fontsize=12, color='r')
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
            
        plt.show()


# ORC Processing
# print(pytesseract.image_to_string(im, lang='eng'))
