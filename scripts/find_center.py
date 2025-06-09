import sys
import numpy as np
import copy

work_dir = "/sf/maloja/data/p19750/work/scripts"
sys.path.append(work_dir)
import parameters

# Careful, the default center is not shape//2
default_center = np.array([1155, 1105])

ts_y = np.array(parameters.tile_start_y)
ts_x = np.array(parameters.tile_start_x)
size_y = parameters.tile_size_y
size_x = parameters.tile_size_x

# choose the 3rd and 6th tile for center finding 
tile_nums = [2, 5]

# Minimize the asymmetry to find the center of JungFrau image
def Auto_Find_Center(JF_Img, mask, tile_start_y = ts_y[tile_nums], tile_start_x = ts_x[tile_nums], tile_size_y = size_y, tile_size_x = size_x, size = 300, sliding_steps = 61, delta = 20): # JF_Img better be background subtracted and after common mode correction

    # mask: 1 means invalid while 0 means valid. valid_mask: 1 means valid while 0 means invalid
    valid_mask = ~mask
    New_Img = copy.deepcopy(JF_Img)
    # Extract the two available tiles (adjust these coordinates based on your tiles)
    y1_start = tile_start_y[0]+tile_size_y-size-delta
    x1_start = tile_start_x[0]+tile_size_x-size-delta
    y2_start = tile_start_y[1] + delta
    x2_start = tile_start_x[1] + delta
    
    Original_Region1 = New_Img[y1_start:y1_start+size, x1_start:x1_start+size]
    Original_Region2 = New_Img[y2_start:y2_start+size, x2_start:x2_start+size]
    OR1_mask = valid_mask[y1_start:y1_start+size, x1_start:x1_start+size]
    OR2_mask = valid_mask[y2_start:y2_start+size, x2_start:x2_start+size]

    Region1_flipped = np.flip(Original_Region1)
    Region2_flipped = np.flip(Original_Region2)
    R1f_mask = np.flip(OR1_mask)
    R2f_mask = np.flip(OR2_mask)

    # The idea is trying to search for the cropped flipped region1 in original region2, vice versa
    cropped_flipped_region1 = Region1_flipped[sliding_steps//2:size-sliding_steps//2,sliding_steps//2:size-sliding_steps//2]
    cropped_flipped_region2 = Region2_flipped[sliding_steps//2:size-sliding_steps//2,sliding_steps//2:size-sliding_steps//2]
    cfr1_mask = R1f_mask[sliding_steps//2:size-sliding_steps//2,sliding_steps//2:size-sliding_steps//2]
    cfr2_mask = R2f_mask[sliding_steps//2:size-sliding_steps//2,sliding_steps//2:size-sliding_steps//2]

    a, b = cropped_flipped_region1.shape
    c, d = Original_Region1.shape
    asymmetry = np.zeros((c - a + 1, d - b + 1))

    for i in range(c - a + 1):  # Rows
        for j in range(d - b + 1):  # Columns
            
            # Extract the sliding window
            sliding_window1 = Original_Region1[i:i + a, j:j + b]
            sliding_window2 = Original_Region2[i:i + a, j:j + b]
            sliding_window = np.vstack((sliding_window1, sliding_window2))

            sw1_mask = OR1_mask[i:i + a, j:j + b]
            sw2_mask = OR2_mask[i:i + a, j:j + b]
            sw_mask = np.vstack((sw1_mask, sw2_mask))
            
            ref_image = np.vstack((cropped_flipped_region2, cropped_flipped_region1))
            rfi_mask = np.vstack((cfr2_mask, cfr1_mask))

            common_mask = sw_mask&rfi_mask

            # # Normalize the sliding window
            # sliding_window_normalized = normalize(sliding_window[common_mask])
            # ref_image_normalized = normalize(ref_image[common_mask])
            # # print(ref_image_normalized.shape)
            
            
            # Flattern the sliding window
            sliding_window_flatterned = sliding_window[common_mask]
            ref_image_flatterned = ref_image[common_mask]
            # print(ref_image_normalized.shape)
            
            # Compute the asymmetry between the regions
            num_pixels_involved = ref_image_flatterned.shape[0]
            # if num_pixels_involved == 0:
            #     print(f'divide by zero problem.')
            asymmetry[i, j] = np.sum((ref_image_flatterned - sliding_window_flatterned)**2)/num_pixels_involved
    
    # Find the peak of the cross-asymmetry

    y_shift, x_shift = np.unravel_index(np.argmin(asymmetry), asymmetry.shape)
    center_shift_y = (y_shift - asymmetry.shape[0] // 2)//2
    center_shift_x = (x_shift - asymmetry.shape[1] // 2)//2
    return asymmetry, np.array([center_shift_y, center_shift_x])

tile_nums4 = [2, 5, 4, 3]

def Auto_Find_Center_4(JF_Img, mask, tile_start_y = ts_y[tile_nums4], tile_start_x = ts_x[tile_nums4], tile_size_y = size_y, tile_size_x = size_x, size = 300, sliding_steps = 61, delta = 20): # JF_Img better be background subtracted and after common mode correction

    # mask: 1 means invalid while 0 means valid. valid_mask: 1 means valid while 0 means invalid
    valid_mask = ~mask
    New_Img = copy.deepcopy(JF_Img)
    # Extract the two available tiles (adjust these coordinates based on your tiles)
    y1_start = tile_start_y[0]+tile_size_y-size-delta
    x1_start = tile_start_x[0]+tile_size_x-size-delta
    y2_start = tile_start_y[1] + delta
    x2_start = tile_start_x[1] + delta

    y3_start = tile_start_y[2]+delta
    x3_start = tile_start_x[2]+tile_size_x-size-delta
    y4_start = tile_start_y[3]+tile_size_y-size-delta
    x4_start = tile_start_x[3] + delta
    
    Original_Region1 = New_Img[y1_start:y1_start+size, x1_start:x1_start+size]
    Original_Region2 = New_Img[y2_start:y2_start+size, x2_start:x2_start+size]
    OR1_mask = valid_mask[y1_start:y1_start+size, x1_start:x1_start+size]
    OR2_mask = valid_mask[y2_start:y2_start+size, x2_start:x2_start+size]

    Original_Region3 = New_Img[y3_start:y3_start+size, x3_start:x3_start+size]
    Original_Region4 = New_Img[y4_start:y4_start+size, x4_start:x4_start+size]
    OR3_mask = valid_mask[y3_start:y3_start+size, x3_start:x3_start+size]
    OR4_mask = valid_mask[y4_start:y4_start+size, x4_start:x4_start+size]    

    Region1_flipped = np.flip(Original_Region1)
    Region2_flipped = np.flip(Original_Region2)
    R1f_mask = np.flip(OR1_mask)
    R2f_mask = np.flip(OR2_mask)

    Region3_flipped = np.flip(Original_Region3)
    Region4_flipped = np.flip(Original_Region4)
    R3f_mask = np.flip(OR3_mask)
    R4f_mask = np.flip(OR4_mask)

    # The idea is trying to search for the cropped flipped region1 in original region2, vice versa
    cropped_flipped_region1 = Region1_flipped[sliding_steps//2:size-sliding_steps//2,sliding_steps//2:size-sliding_steps//2]
    cropped_flipped_region2 = Region2_flipped[sliding_steps//2:size-sliding_steps//2,sliding_steps//2:size-sliding_steps//2]
    cfr1_mask = R1f_mask[sliding_steps//2:size-sliding_steps//2,sliding_steps//2:size-sliding_steps//2]
    cfr2_mask = R2f_mask[sliding_steps//2:size-sliding_steps//2,sliding_steps//2:size-sliding_steps//2]

    cropped_flipped_region3 = Region3_flipped[sliding_steps//2:size-sliding_steps//2,sliding_steps//2:size-sliding_steps//2]
    cropped_flipped_region4 = Region4_flipped[sliding_steps//2:size-sliding_steps//2,sliding_steps//2:size-sliding_steps//2]
    cfr3_mask = R3f_mask[sliding_steps//2:size-sliding_steps//2,sliding_steps//2:size-sliding_steps//2]
    cfr4_mask = R4f_mask[sliding_steps//2:size-sliding_steps//2,sliding_steps//2:size-sliding_steps//2]

    a, b = cropped_flipped_region1.shape
    c, d = Original_Region1.shape
    asymmetry = np.zeros((c - a + 1, d - b + 1))

    for i in range(c - a + 1):  # Rows
        for j in range(d - b + 1):  # Columns
            
            # Extract the sliding window
            sliding_window1 = Original_Region1[i:i + a, j:j + b]
            sliding_window2 = Original_Region2[i:i + a, j:j + b]
            sliding_window3 = Original_Region3[i:i + a, j:j + b]
            sliding_window4 = Original_Region4[i:i + a, j:j + b]
            
            sliding_window = np.vstack((sliding_window1, sliding_window2, sliding_window3, sliding_window4))

            sw1_mask = OR1_mask[i:i + a, j:j + b]
            sw2_mask = OR2_mask[i:i + a, j:j + b]
            sw3_mask = OR3_mask[i:i + a, j:j + b]
            sw4_mask = OR4_mask[i:i + a, j:j + b]
            sw_mask = np.vstack((sw1_mask, sw2_mask, sw3_mask, sw4_mask))
            
            ref_image = np.vstack((cropped_flipped_region2, cropped_flipped_region1, cropped_flipped_region4, cropped_flipped_region3))
            rfi_mask = np.vstack((cfr2_mask, cfr1_mask, cfr4_mask, cfr3_mask))

            common_mask = sw_mask&rfi_mask

            # # Normalize the sliding window
            # sliding_window_normalized = normalize(sliding_window[common_mask])
            # ref_image_normalized = normalize(ref_image[common_mask])
            # # print(ref_image_normalized.shape)
            
            
            # Flattern the sliding window
            sliding_window_flatterned = sliding_window[common_mask]
            ref_image_flatterned = ref_image[common_mask]
            # print(ref_image_normalized.shape)
            
            # Compute the asymmetry between the regions
            num_pixels_involved = ref_image_flatterned.shape[0]
            asymmetry[i, j] = np.sum((ref_image_flatterned - sliding_window_flatterned)**2)/num_pixels_involved
    
    # Find the peak of the cross-asymmetry

    y_shift, x_shift = np.unravel_index(np.argmin(asymmetry), asymmetry.shape)
    center_shift_y = (y_shift - asymmetry.shape[0] // 2)//2
    center_shift_x = (x_shift - asymmetry.shape[1] // 2)//2
    return asymmetry, np.array([center_shift_y, center_shift_x])
