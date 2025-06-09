import numpy as np
import copy
# this is only for pilot data with a missing 2nd quarter, we use the 3rd quarter to fill the 2nd quarter

def fill_quarter(jf_img0, mask, center = None, tile_starts = np.array([[1100, 0], [1649, 0]]), size_y = 512, size_x = 1024):

    jf_img = copy.deepcopy(jf_img0)
    y1, x1 = tile_starts[0]
    y2, x2 = tile_starts[1]
    
    tile1 = jf_img[y1:y1+size_y, x1:x1+size_x]
    tile2 = jf_img[y2:y2+size_y, x2:x2+size_x]

    # Rotate both tiles by 180 degrees
    tile1_rotated = np.rot90(tile1, k=2)  # 180 degrees
    tile2_rotated = np.rot90(tile2, k=2)  # 180 degrees
    if mask is None:
        mask = np.zeros_like(jf_img0)
    new_mask = copy.deepcopy(mask)
    if center is None:
        center = np.array(jf_img0.shape)//2
    center_y, center_x = center

    # New quarters
                 
    y1p_start = max(0, 2*center_y - (y1 + size_y))
    y1p_end = min(jf_img.shape[0], 2*center_y - y1)
    x1p_start = max(0, 2*center_x - (x1 + size_x))
    x1p_end = min(jf_img.shape[1], 2*center_x - x1)
    length_y1p, length_x1p = y1p_end - y1p_start, x1p_end - x1p_start
    y2p_start = max(0, 2*center_y - (y2 + size_y))
    y2p_end = min(jf_img.shape[0], 2*center_y - y2)
    x2p_start = max(0, 2*center_x - (x2 + size_x))
    x2p_end = min(jf_img.shape[1], 2*center_x - x2)
    length_y2p, length_x2p = y2p_end - y2p_start, x2p_end - x2p_start    
    jf_img[y1p_start:y1p_end, x1p_start:x1p_end] = jf_img[y1+size_y-length_y1p:y1+size_y, x1+size_x-length_x1p:x1+size_x][::-1, ::-1]
    jf_img[y2p_start:y2p_end, x2p_start:x2p_end] = jf_img[y2+size_y-length_y2p:y2+size_y, x2+size_x-length_x2p:x2+size_x][::-1, ::-1]

    new_mask[y1p_start:y1p_end, x1p_start:x1p_end] = mask[y1+size_y-length_y1p:y1+size_y, x1+size_x-length_x1p:x1+size_x][::-1, ::-1]
    new_mask[y2p_start:y2p_end, x2p_start:x2p_end] = mask[y2+size_y-length_y2p:y2+size_y, x2+size_x-length_x2p:x2+size_x][::-1, ::-1]

    return jf_img, new_mask