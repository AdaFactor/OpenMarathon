import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog, match_template
from skimage import data, exposure, io

image = io.imread('../data/001.jpg', as_grey=True)
tag = io.imread('../data/template.jpg', as_grey=True)

result = match_template(image, tag)
ij = np.unravel_index(np.argmax(result), result.shape)
x, y = ij[::-1]
cx, cy = result.shape

fig = plt.figure(figsize=(8, 3))
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

cut_tag = image[y:y+htag, x:x+wtag]
ax3.imshow(cut_tag, cmap=plt.cm.gray)
ax3.set_axis_off()
ax3.set_title('cut_tag')

plt.show()