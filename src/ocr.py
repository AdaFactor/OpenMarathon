import os
import sys
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from face_detection import cropped_images
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, gaussian, frangi
from skimage.measure import find_contours

def ada_ocr(image_path=None, debug=False):
    # Initialization
    predicted_results = []

    # Import Regions of interest
    cropped = cropped_images(image_path=image_path)

    # Prepoceesing
    for crop in cropped:
        # Converting RGB to GRAY
        gray_image = rgb2gray(crop)
        
        # Find Global Otsu
        threshold_global_otsu = threshold_otsu(gray_image)
        global_otsu = gray_image >= threshold_global_otsu

        # Take Filters
        im = gaussian(global_otsu, sigma=1)
        im = frangi(global_otsu)

        # Find Contours 
        contours = find_contours(im, 1e-8)
        contours_lens = np.sort([ len(c) for c in contours ])

        # Calculate boundary for select contours
        lower_bound = np.mean(contours_lens)
        upper_bound = contours_lens[-5]
        
        # Selected contours by boundery
        contours = [ c for c in contours if len(c[:, 0]) > lower_bound and len(c[:, 0]) < upper_bound]

    # Working with Contour regions       
        for n, contour in enumerate(contours):
            # Find Left, Top, Right, Bottom for each contour region
            min_x, min_y = int(min(contour[:, 1])), int(min(contour[:, 0])) # left, top
            max_x, max_y = int(max(contour[:, 1])), int(max(contour[:, 0])) # right, bottom
            width = max_x - min_x
            heigh = max_y - min_y

            # Crop ROI from contour and converting to Binary image
            candidate = gray_image[min_y:min_y+heigh, min_x:min_x+width]
            threshold = threshold_otsu(candidate)
            candidate = candidate >= threshold
            
            # Saving Image
            plt.axis('off')
            plt.imshow(candidate, cmap=plt.cm.gray)
            plt.savefig('../data/temp.png')

    # OCR Processing
            test_img = Image.open('../data/temp.png')
            predict_text = pytesseract.image_to_string(test_img)
            if len(predict_text) > 0:
                predicted_results += [predict_text]
                
    return predicted_results

def main():
    image_path = sys.argv[1]
    result = ada_ocr(image_path=image_path)
    print(result)

if __name__ == '__main__':
    main()