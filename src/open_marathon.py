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
# tag = io.imread('/'.join([data_dir, 'template_0.jpg']), as_grey=True)

fig = plt.figure(figsize=(20, 5))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2, adjustable='box-forced')
ax3 = plt.subplot(1, 3, 3)

tag_templates = []
globel_max_peak = []
max_xy = []
edgecolor = ['r', 'b', 'g', 'w']

num_tags_templates = 4
for i in np.arange(0, num_tags_templates):
    tag_templates += ['/'.join([data_dir, 'template_'+ str(i) +'.jpg'])]
    tag_temp = io.imread(tag_templates[i], as_grey=True)
    htag, wtag = tag_temp.shape
    result = match_template(image, tag_temp)
    globel_max_peak += [ np.unravel_index(np.argmax(result), result.shape) ]
    max_x, max_y = globel_max_peak[i][::-1]
    rect = plt.Rectangle((max_x, max_y), wtag, htag, edgecolor=edgecolor[i], facecolor='none')
    ax2.add_patch(rect)

ax1.imshow(tag_temp, cmap=plt.cm.gray)
ax1.set_axis_off()
ax1.set_title('template')

ax2.imshow(image, cmap=plt.cm.gray)
ax2.set_axis_off()
ax2.set_title('image')
# highlight matched region
# print('max:',max_x, max_y)
# local_peak = peak_local_max(result, min_distance=200)
# htag, wtag = tag.shape
# for peak in local_peak:
#     x, y = peak[::-1]
#     print(x,y)
#     rect = plt.Rectangle((x, y), wtag, htag, edgecolor='r', facecolor='none')
#     ax2.add_patch(rect)

# rect = plt.Rectangle((max_x, max_y), wtag, htag, edgecolor='g', facecolor='none')
# ax2.add_patch(rect)

cut_tag = image[max_y:max_y+htag, max_x:max_x+wtag]
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
