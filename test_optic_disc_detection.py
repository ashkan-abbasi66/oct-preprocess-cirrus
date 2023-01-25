import cv2

import os
import numpy as np
import matplotlib.pyplot as plt
import time

from scipy import ndimage as nd

from read_img import read_img
from detect_retina import detect_retina

from utils import *

"""
read_img
"""
data_dir = r'C:\Users\abbasi\POSTDOC\PYTHON_CODES\oct-3d-ohsu\nyudata_analysis\samples'
# filepath = os.path.join(data_dir, '330_Optic Disc Cube 200x200_2-21-2012_9-8-0_OD_sn12108_cube_z.img')
filepath = os.path.join(data_dir, '330_Optic Disc Cube 200x200_2-21-2012_9-8-0_OD_sn12108_cube_z.img')
filepath = os.path.join(data_dir, '2468_Optic Disc Cube 200x200_5-26-2011_10-37-0_OD_sn104459_cube_z.img')
img_vol = read_img(filepath)

"""

"""
pixelsX=200
numFrames=200
pixelsZ=1024
xWidth=6
yWidth=6
zWidth=2

xScale = xWidth * 1000 / pixelsX
yScale = yWidth * 1000 / numFrames
zScale = zWidth * 1000 / pixelsZ

params = dict()
params['sigma_lat'] = 2 * 16.67 #*2   # --------------
params['sigma_ax'] = 0.5 * 11.6 #*2   # --------------

# params['maxdist'] = 386.73  # ~100 pixels in spectralis
# maxdist = params['maxdist']  # maximum distance from ILM to ISOS
# maxdist_bm = 116.02  # maximum distance from ISOS to BM
# isosThresh = 20  # Minimum distance from ISOS to BM

sigma_lat = params['sigma_lat'] / (xScale)
sigma_ax = params['sigma_ax'] / (zScale)

sigma_ax = float(sigma_ax)
sigma_lat = float(sigma_lat)
grad = nd.gaussian_filter(img_vol, sigma=(sigma_ax, 0, 1), mode='nearest',
                          order=0, truncate=2 * np.round(2 * sigma_ax) + 1)
grad = nd.gaussian_filter(grad, sigma=(0, sigma_lat, 1), mode='nearest',
                          order=0, truncate=2 * np.round(2 * sigma_lat) + 1)
img_denoised = grad

grad = (grad - grad.min()) / (grad.max() - grad.min())

# compute gradient along Z direction
grad = nd.sobel(grad, mode='nearest', axis=0)

grad_o = grad.copy()

max1pos = np.argmax(grad, axis=0)
show_image(max1pos,title="locations of maximum gradients")

# show_sample(img_vol,slice=100,title="a sample")


max1 = np.max(grad, axis=0)

max1 = (max1 - max1.min()) / (max1.max() - max1.min())
max1 = max1.astype(np.uint8)

# Mark the location of maximum gradient
max1pos[0:50,:] = 0
max1pos[-50:-1,:] = 0
x0,y0 = mark_max_location(max1pos, img_denoised.mean(0),show=False)

# k = 15
# max2pos = max1pos.copy()
# max2pos[x0-k:x0+k,:] = 0
# max2pos[:,y0-k:y0+k] = 0
# x1,y1 = mark_max_location(max2pos, img_denoised.mean(0),show=False)

# x_percent =  0.85
x_percent =  0.60

out_ = max1pos.copy()
for i in range(max1pos.shape[0]):
    for j in range(max1pos.shape[1]):
        if out_[i,j]<= x_percent*max1pos[x0,y0]:
            out_[i,j] = 0

show_image(out_,"keep x% of maximum",show=False)
show_image(img_vol.mean(0),"original",show=True)


square = nd.generate_binary_structure(rank=2, connectivity=21)
out_2 = nd.white_tophat(input=out_, structure=square)
show_image(out_2,"after whitehat")


from skimage.filters import threshold_otsu
thresh = threshold_otsu(out_)
binary = out_ > thresh

show_image(binary,"binary")

indices = np.nonzero(binary)
xy = np.unravel_index(indices, binary.shape, 'C')

s = np.zeros([1,2])
n = xy[1].shape[1]
for j in range(n):
    s = xy[1][:,j] + s
mvector = s/n
mvector = np.floor(mvector)
mvector = mvector[0]

# fuse with the maximum location
mvector = (mvector + np.array([x0,y0]).reshape(1,2))/2
mvector = np.round(mvector)[0].astype(np.int)

fig, ax = plt.subplots(1)
plt.imshow(img_vol.mean(0))
ax.add_patch(Circle((mvector[0], mvector[1]), radius=2, color='red'))
ax.add_patch(Circle((mvector[0], mvector[1]), radius=70, color='red',fill=False))
plt.title("mean point")
plt.show()
# out_2 = nd.black_tophat(input=out_, structure=square)
# show_image(out_2,"after blackhat",show=True)





# const = 50
# step = 5
# xrange = np.arange(x0-const,x0+const, step)
# yrange = range(y0-const,y0+const,step)
#
# outmax = max1.copy()
# for i in range(max1.shape[0]):
#     for j in range(max1.shape[1]):
#         if (i not in xrange) or (j not in yrange):
#             outmax[i,j] = 0
# show_image(outmax,"keep x% pixels")

# """
# keep x% of pixels with highest values around the maximum location
# """
# t = np.sort(max1pos.reshape([-1,1]))
# ind = np.round(len(t)*0.8).astype(np.int)
# t[:ind] = 0
# t = t.reshape([max1pos.shape[0],max1pos.shape[1]])
# show_image(t,"keep x% pixels")


"""
ddd
"""
# markers = np.ones_like(max1pos).astype(np.int16)
# markers[x0,y0] = 2
#
#
# square = nd.generate_binary_structure(rank=2, connectivity=11)
# t = nd.watershed_ift(max1pos.astype(np.uint8), markers, structure=square)
# show_image(t)
# mark_max_location(t, img_denoised.mean(0))



# right
# max1pos[max1pos<=400]=0

# show_image(t)
# mark_max_location(t, img_denoised.mean(0))




"""
Hough Transform
"""
# image = max1pos
# image = (image - image.min()) / (image.max() - image.min())
# image = (image*255.0).astype(np.uint8)
# circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1.2, 100)
# output = image.copy()
#
# # ensure at least some circles were found
# if circles is not None:
# 	# convert the (x, y) coordinates and radius of the circles to integers
# 	circles = np.round(circles[0, :]).astype("int")
# 	# loop over the (x, y) coordinates and radius of the circles
# 	for (x, y, r) in circles:
# 		# draw the circle in the output image, then draw a rectangle
# 		# corresponding to the center of the circle
# 		cv2.circle(output, (x, y), r, (0, 255, 0), 4)
# 		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
# 	# show the output image
# 	cv2.imshow("output", np.hstack([image, output]))
# 	cv2.waitKey(0)
# else:
#     print("There is no circle!")