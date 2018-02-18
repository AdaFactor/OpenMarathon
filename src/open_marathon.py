import sys
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

from skimage.feature import hog, match_template
from skimage.filters import threshold_otsu
from skimage import data, exposure, io
from PIL import Image

image = io.imread('../data/'+ sys.argv[1] +'.jpg', as_grey=True)
tag = io.imread('../data/template.jpg', as_grey=True)

result = match_template(image, tag)
ij = np.unravel_index(np.argmax(result), result.shape)
x, y = ij[::-1]

fig = plt.figure(figsize=(20, 3))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2, adjustable='box-forced')
ax3 = plt.subplot(1, 3, 3)

ax1.imshow(tag, cmap=plt.cm.gray)
ax1.set_axis_off()
ax1.set_title('template')

ax2.imshow(image, cmap=plt.cm.gray)
ax2.set_axis_off()
ax2.set_title('image')
# highlight matched region
htag, wtag = tag.shape
rect = plt.Rectangle((x, y), wtag, htag, edgecolor='r', facecolor='none')
ax2.add_patch(rect)

cut_tag = image[y:y+htag, x+100:x+wtag-30]
block_size = 35
adaptive_thresh = threshold_otsu(cut_tag)
binary_adaptive = cut_tag > adaptive_thresh

io.imsave('./test_tag.jpg', binary_adaptive)
raw_text = pytesseract.image_to_string(Image.open('./test_tag.jpg'))
filted_text = ''.join(filter(lambda x: x.isdigit(), raw_text)) 
print('OCR Extracting:', filted_text)

ax3.imshow(binary_adaptive, cmap=plt.cm.gray)
ax3.set_axis_off()
ax3.set_title('cut_tag')

plt.show()