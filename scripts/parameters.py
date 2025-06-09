import numpy as np

# Geometry of the JungFrau detector
tile_start_y=[0, 149, 550, 699, 1100, 1249, 1649, 1798]
tile_start_x=[149, 1185, 149, 1185, 0, 1037, 0, 1037]
tile_size_x = 1024
tile_size_y = 512

pixel_size = 75e-6 # 75um
det_samp_dist = 340e-3 # 340mm

img_shape = (2312, 2215)

Quarter1, Quarter2, Quarter3, Quarter4 = np.zeros(img_shape, dtype = 'bool'), np.zeros(img_shape, dtype = 'bool'), np.zeros(img_shape, dtype = 'bool'), np.zeros(img_shape, dtype = 'bool')

for i in [0, 2]:
    Quarter1[tile_start_y[i]:tile_start_y[i]+tile_size_y,tile_start_x[i]:tile_start_x[i]+tile_size_x] = 1
for i in [1, 3]:
    Quarter2[tile_start_y[i]:tile_start_y[i]+tile_size_y,tile_start_x[i]:tile_start_x[i]+tile_size_x] = 1
for i in [4, 6]:
    Quarter3[tile_start_y[i]:tile_start_y[i]+tile_size_y,tile_start_x[i]:tile_start_x[i]+tile_size_x] = 1
for i in [5, 7]:
    Quarter4[tile_start_y[i]:tile_start_y[i]+tile_size_y,tile_start_x[i]:tile_start_x[i]+tile_size_x] = 1