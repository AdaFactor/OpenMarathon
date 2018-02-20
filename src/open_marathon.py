import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

from skimage.feature import hog, match_template, peak_local_max
from skimage.filters import threshold_otsu
from skimage import data, exposure, io
from PIL import Image

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = '/'.join([ root_dir, 'data'])
image = io.imread('/'.join([data_dir, sys.argv[1]]), as_grey=True)
tag = io.imread('/'.join([data_dir, 'template_4.jpg']), as_grey=True)

result = match_template(image, tag)

ij = np.unravel_index(np.argmax(result), result.shape)
max_x, max_y = ij[::-1]

fig = plt.figure(figsize=(20, 5))
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
print('max:',max_x, max_y)
local_peak = peak_local_max(result, min_distance=200)
htag, wtag = tag.shape
for peak in local_peak:
    x, y = peak[::-1]
    print(x,y)
    rect = plt.Rectangle((x, y), wtag, htag, edgecolor='r', facecolor='none')
    ax2.add_patch(rect)

rect = plt.Rectangle((max_x, max_y), wtag, htag, edgecolor='g', facecolor='none')
ax2.add_patch(rect)

cut_tag = image[max_y:max_y+htag, max_x+100:max_x+wtag-30]
block_size = 35
adaptive_thresh = threshold_otsu(cut_tag)
binary_adaptive = cut_tag > adaptive_thresh

test_data_dir = '/'.join([data_dir, 'test_tag.jpg']) 
io.imsave(test_data_dir, binary_adaptive)
raw_text = pytesseract.image_to_string(Image.open(test_data_dir))
filted_text = ''.join(filter(lambda x: x.isdigit(), raw_text)) 
print('OCR Extracting:', filted_text)

ax3.imshow(binary_adaptive, cmap=plt.cm.gray)
ax3.set_axis_off()
ax3.set_title('cut_tag')

plt.show()