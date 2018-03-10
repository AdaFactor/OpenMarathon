import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

from skimage.feature import hog, match_template, peak_local_max
from skimage.filters import threshold_otsu
from skimage.transform import pyramid_reduce
from skimage import data, exposure, io
from PIL import Image

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = '/'.join([root_dir, 'data'])
image = io.imread('/'.join([data_dir, sys.argv[1]]), as_grey=True)
image = pyramid_reduce(image, downscale=4)
# tag = io.imread('/'.join([data_dir, 'template_0.jpg']), as_grey=True)

fig = plt.figure(figsize=(20, 5))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2, adjustable='box-forced')
ax3 = plt.subplot(1, 3, 3)

tag_templates = []
globel_max_peak = []
max_xy = []
tags_shape = []
edgecolor = ['r', 'b', 'g', 'w']

num_tags_templates = 4
for i in np.arange(0, num_tags_templates):
    tag_templates += ['/'.join([data_dir, 'template_' + str(i) + '.jpg'])]
    tag_temp = pyramid_reduce(
        io.imread(tag_templates[i], as_grey=True),
        downscale=4
    )
    htag, wtag = tag_temp.shape
    tags_shape += [(htag, wtag)]
    result = match_template(image, tag_temp)
    globel_max_peak += [np.unravel_index(np.argmax(result), result.shape)]
    max_x, max_y = globel_max_peak[i][::-1]
    max_xy += [(max_x, max_y)]
    rect = plt.Rectangle(
        (max_x, max_y),
        wtag,
        htag,
        edgecolor=edgecolor[i],
        facecolor='none'
    )
    ax2.add_patch(rect)
print(max_xy)
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
max_no = int(sys.argv[2])
x, y = max_xy[max_no][0], max_xy[max_no][1]
h, w = tags_shape[max_no][0], tags_shape[max_no][1]
cut_tag = image[y:y+h, x:x+w]
block_size = 35
adaptive_thresh = threshold_otsu(cut_tag)
binary_adaptive = cut_tag > adaptive_thresh

test_data_dir = '/'.join([data_dir, 'test_tag.jpg'])
io.imsave(test_data_dir, cut_tag)
raw_text = pytesseract.image_to_string(Image.open(test_data_dir))
# filted_text = ''.join(filter(lambda x: x.isdigit(), raw_text))
print('OCR Extracting:', raw_text)

# ax3.imshow(binary_adaptive, cmap=plt.cm.gray)
ax3.imshow(cut_tag, cmap=plt.cm.gray)
ax3.set_axis_off()
ax3.set_title('cut_tag')

plt.show()
