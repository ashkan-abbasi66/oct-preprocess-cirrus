import os
import numpy as np
import matplotlib.pyplot as plt
import time

from read_img import read_img
from detect_retina import detect_retina

from utils import show_sample

"""
read_img
"""
data_dir = r'C:\Users\abbasi\POSTDOC\PYTHON_CODES\oct-3d-ohsu\nyudata_analysis\samples'
filepath = os.path.join(data_dir, '330_Optic Disc Cube 200x200_2-21-2012_9-8-0_OD_sn12108_cube_z.img')
img_vol = read_img(filepath)

print(type(img_vol))   # <class 'numpy.ndarray'>
print(img_vol.shape)   # Z*X*Y

show_sample(img_vol,title='original volume')
plt.show()


"""
detect_retina
"""

# Options
flatten = False
outputRetinaMask = True


start = time.time()
vol_out, retina_mask, upper_bound, lower_bound = detect_retina(img_vol,
                                                                     outputRetinaMask = outputRetinaMask,
                                                                     flatten=flatten)
elapsed_time = time.time() - start
print("Elapsed time %.2f"%elapsed_time)

if vol_out is not None:
    show_sample(vol_out,title='flattened volume')
    plt.show()
else:
    vol_out = img_vol

if retina_mask is not None:
    show_sample(retina_mask,title='output mask')
    plt.show()

    show_sample(np.multiply(vol_out,retina_mask), title='multiplication of mask and volume')
    plt.show()


print("THE END.")