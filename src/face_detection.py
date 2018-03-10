import os
import dlib
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import rescale
from skimage.color import rgb2gray


# Set path
PRO_DIR = os.path.dirname(os.path.abspath('../'+__file__))
DATA_DIR = os.path.join(PRO_DIR, 'data')


def cropped_images(image_path=None, extend_left=50, width_param=2, heigh_param=5, see_full=False):
    # Reading an image
    if image_path == None:
        image_path = os.path.join(DATA_DIR, '001.jpg')
    image = io.imread(image_path)
    
    # Face Detector
    detector = dlib.get_frontal_face_detector()
    dets = detector(image, 1)
    print('Number of faces detected = {}'.format(len(dets)))

    # Initial Cropped list
    cropped = []

    for i, d in enumerate(dets):
        left, top = d.left()-extend_left, d.top()+100
        right, bottom = d.right(), d.bottom()+100
        d_point = (left, top)
        width = (right - left)*width_param
        heigh = (bottom - top)*heigh_param

        cropped += [image[top:top+heigh, left:left+width]]

    # See Full image if see_full=TRUE
    if see_full:
        plt.imshow(image)
        plt.show()

    return cropped


def main():
    # Matplotlib Configurations
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    cropped = cropped_images()
    for crop in cropped:
        plt.imshow(crop, cmap='gray')
        plt.show()
        

if __name__ == '__main__':
    main()
