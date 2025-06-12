import sys
import numpy as np
import copy
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from skimage.restoration import unwrap_phase

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


def find_center_minimize_asymmetry(JF_Img, mask, center_range=40, cropsize=2048):
    """
    Find the center of an image by minimizing asymmetry.

    Parameters
    ----------
    JF_Img : 2D numpy array
        Input image data.
    mask : 2D numpy array or boolean array
        Mask indicating invalid or bad pixels (True or 1 = masked out).
    center_range : int, optional
        Width and height of the search grid (number of shifts tested) around the image center.
    cropsize : int, optional
        Side length of the square crop used for asymmetry calculation (must be even).

    Returns
    -------
    asymmetries : 2D numpy array
        Array of asymmetry metrics for each tested shift.
    center_shift : numpy array of shape (2,)
        Optimal (delta_y, delta_x) shift from the initial image center that minimizes asymmetry.
    """
    # 1) Compute initial guess: geometric center of the full image
    init_center = np.array(JF_Img.shape) // 2
    cy, cx = init_center

    # 2) Prepare storage for asymmetry values
    asymmetries = np.zeros((center_range, center_range))

    # 3) Precompute half ranges for convenience
    half_range = center_range // 2
    half_crop = cropsize // 2

    # 4) Loop over all candidate shifts in y (dy) and x (dx)
    for dy in range(-half_range, half_range):
        for dx in range(-half_range, half_range):
            # 4a) Compute shifted center coordinates
            cy_shifted = cy + dy
            cx_shifted = cx + dx

            # 4b) Crop the image and mask around this candidate center
            img_crop = JF_Img[
                cy_shifted - half_crop : cy_shifted + half_crop,
                cx_shifted - half_crop : cx_shifted + half_crop
            ]
            mask_crop = mask[
                cy_shifted - half_crop : cy_shifted + half_crop,
                cx_shifted - half_crop : cx_shifted + half_crop
            ].astype(bool)

            # 4c) Create flipped versions to enforce symmetry
            img_flip = np.flip(img_crop)
            mask_flip = np.flip(mask_crop)

            # 4d) Combine masks: any pixel masked in original or flipped is invalid
            combined_mask = mask_crop | mask_flip
            valid_pixels = ~combined_mask  # True where data is valid

            # 4e) Compute asymmetry: squared L2 norm of difference, normalized by count
            diff = img_crop * valid_pixels - img_flip * valid_pixels
            asym_val = np.linalg.norm(diff)**2 / np.sum(valid_pixels)

            # 4f) Store this asymmetry metric
            asymmetries[dy + half_range, dx + half_range] = asym_val

    # 5) Identify the shift that gives minimal asymmetry
    flat_idx = asymmetries.argmin()
    best_pos = np.unravel_index(flat_idx, asymmetries.shape)
    center_shift = np.array([
        best_pos[0] - half_range,
        best_pos[1] - half_range
    ])

    return asymmetries, center_shift
            
    
def center_of_mass_method(JF_Img, mask, center_guess, cropsize=2048):
    """
    Compute the shift of the center-of-mass of the image intensities
    (masked for symmetry) relative to the given center_guess.

    Parameters
    ----------
    JF_Img : 2D ndarray
        The full image.
    mask : 2D ndarray of bool or int
        A mask to exclude bad pixels (1 = mask out, 0 = good).
    center_guess : tuple of (cy, cx)
        Initial guess for the center in image coordinates.
    cropsize : int
        Side length of the square crop (must be even).

    Returns
    -------
    center_shift : ndarray (delta_y, delta_x) giving how far the true center-of-mass
        is from the center of the crop.
    """
    cy, cx = center_guess
    cy, cx = int(round(cy)), int(round(cx))
    half = cropsize // 2

    # 1) Crop image and mask
    img_crop  = JF_Img[cy-half:cy+half, cx-half:cx+half]
    img_flip = np.flip(img_crop)
    msk_crop  = mask[cy-half:cy+half, cx-half:cx+half].astype(bool)
    msk_flip  = np.flip(msk_crop)
    img_combine = img_crop*(~msk_crop) + img_flip*(~msk_flip)*msk_crop
    # 2) Enforce symmetry: flip mask and OR
    
    # final_mask = msk_crop | msk_flip

    # valid pixels = those *not* masked out
    # valid_mask = ~final_mask
    valid_mask = (~msk_crop) | (~msk_flip)

    # 3) Compute total mass and COM in crop coords
    # weights = img_crop * valid_mask
    weights = img_combine * valid_mask
    total_mass = weights.sum()
    if total_mass == 0:
        # avoid division by zero
        print('total mass is zero')
        return (0.0, 0.0)

    # create coordinate grids of shape (cropsize, cropsize)
    ys, xs = np.indices((cropsize, cropsize))

    y_com = (weights * ys).sum() / total_mass
    x_com = (weights * xs).sum() / total_mass

    # 4) shift relative to the crop center (which is at half,half)
    delta_y = y_com - (cropsize-1)/2
    delta_x = x_com - (cropsize-1)/2

    return np.array([delta_y, delta_x])

# def iterative_center_of_mass(JF_Img, mask, init_center_guess = None, cropsize=2048):
#     if init_center_guess is None:
#         init_center_guess = np.array(JF_Img)/2
#     center = init_center_guess

#     centers = [center]
#     iterative_times = 0
#     while iterative_times < 50:
#         iterative_times += 1
#         center_shift = center_of_mass_method(JF_Img, mask, center, cropsize)
#         cy_shift, cx_shift = center_shift
#         if cy_shift < 1 and cx_shift < 1:
#             return center, centers
#         else: # update center
#             center = center + center_shift
#             centers.append(center) # for debug
#     print('Cannot decide the center')
#     return None, centers


def iterative_center_of_mass(JF_Img, mask, init_center_guess=None, cropsize=2048):
    """
    Iteratively refine the center estimate using a center-of-mass method.

    Parameters
    ----------
    JF_Img : 2D numpy array
        Input image data.
    mask : 2D array or boolean mask
        Mask indicating invalid pixels (True or 1 = masked out).
    init_center_guess : sequence of two floats or ints, optional
        Initial (y, x) center guess in image coordinates. Defaults to geometric center.
    cropsize : int, optional
        Size of square region for center-of-mass calculation (must be even).

    Returns
    -------
    center : array of two floats
        Final (y, x) center estimate after convergence.
    centers : list of arrays
        History of center estimates at each iteration (for debugging or analysis).
    """
    # 1) Set initial center guess to image center if none provided
    if init_center_guess is None:
        # geometric center of the full image
        init_center_guess = np.array(JF_Img.shape) / 2
    center = np.array(init_center_guess, dtype=float)

    # 2) Track the sequence of center estimates
    centers = [center.copy()]
    max_iterations = 50
    iteration = 0

    # 3) Loop until convergence or max iterations
    while iteration < max_iterations:
        iteration += 1
        # 3a) Compute shift via center-of-mass method
        cy_shift, cx_shift = center_of_mass_method(JF_Img, mask, center, cropsize)

        # 3b) Check for convergence: shifts below 1 pixel in both directions
        if abs(cy_shift) < 0.5 and abs(cx_shift) < 0.5:
            # Converged: return final center and history
            return center, centers

        # 3c) Update center estimate, 0.8 to help converge
        # center += 0.8*np.array([cy_shift, cx_shift])
        if cy_shift**2+cx_shift**2 < 1:
            center += np.array([cy_shift, cx_shift])
        else:
            center += np.array([cy_shift/np.sqrt(cy_shift**2+cx_shift**2), cx_shift/np.sqrt(cy_shift**2+cx_shift**2)])

        # 3d) boundary check, the center won't be too far away from the detector center
        center[0] = min(center[0], init_center_guess[0]+15)
        center[0] = max(center[0], init_center_guess[0]-15)
        center[1] = min(center[1], init_center_guess[1]+15)
        center[1] = max(center[1], init_center_guess[1]-15)
        
        centers.append(center.copy())

    # 4) If we exit loop without convergence, warn and return None
    print('Warning: iterative_center_of_mass did not converge after', max_iterations, 'iterations')
    return None, centers



 
from scipy.ndimage import shift as ndi_shift
from skimage.transform import rotate
from skimage.registration import phase_cross_correlation

# Currently, I apply this phase correlation method to find the center of the image, and iterative method not applied

def phase_correlation_method(
    JF_Img: np.ndarray,
    init_center: tuple[float, float],
    mask: np.ndarray,
    cropsize: int,
    upsample_factor: int = 10
):
    """
    Refine beam-center by 180° rotation + phase-correlation,
    properly handling masked (invalid) pixels.

    Parameters
    ----------
    JFImg : (H, W) array
        Diffraction frame; invalid pixels (e.g., masked quadrant) can be any value.
    init_center : (y0, x0)
        Initial guess for the beam center (row, col).
    mask : (H, W) bool array
        True where JFImg pixels are valid; False for masked/invalid pixels.
    upsample_factor : int
        Upsampling for subpixel precision.

    Returns
    -------
    (y0_ref, x0_ref)
        Refined center coordinates (row, col).
    """


    cy, cx = init_center
    cy, cx = int(round(cy)), int(round(cx))
    half = cropsize // 2

    # 1) Crop image and mask, flip them, combine the cropped and flipped to create centrosymmetry
    # I do this try to cancel the insymmetry caused by invalid pixels
    img_crop  = JF_Img[cy-half:cy+half, cx-half:cx+half]
    img_flip = np.flip(img_crop)
    msk_crop  = mask[cy-half:cy+half, cx-half:cx+half].astype(bool)
    msk_flip  = np.flip(msk_crop)
    # img_combine = img_crop*(~msk_crop) + img_flip*(~msk_flip)*msk_crop
    # valid_mask = (~msk_crop) | (~msk_flip)

    # This avoids asymmetry created because of the mask
    img_combine = img_crop*(~(msk_crop|msk_flip))
    valid_mask = ~(msk_crop|msk_flip)
    
    
    # 2) Rotate by 180° about the array center (pivot = init_center in original)
    img_rot = rotate(img_combine, 180, resize=False,
                     order=1, mode='constant', cval=0, preserve_range=True)
    mask_rot = rotate(valid_mask.astype(float), 180, resize=False,
                      order=0, mode='constant', cval=0, preserve_range=True) > 0.5

    # 3) Phase-correlation with masks
    shift_estimate, error, _ = phase_cross_correlation(
        img_combine, img_rot,
        upsample_factor=upsample_factor,
        reference_mask=valid_mask,
        moving_mask=mask_rot
    )
    dy, dx = shift_estimate

    # 4) Correct for 2x effect of pivot error
    y0_ref = cy + dy / 2.0
    x0_ref = cx + dx / 2.0

    return np.array([y0_ref, x0_ref])


# def iterative_phase_correlation(JF_Img, mask, init_center_guess=None, cropsize=2048):
    
#     # 1) Set initial center guess to image center if none provided
#     if init_center_guess is None:
#         # geometric center of the full image
#         init_center_guess = np.array(JF_Img.shape) / 2
#     center = np.array(init_center_guess, dtype=float)

#     # 2) Track the sequence of center estimates
#     centers = [center.copy()]
#     max_iterations = 8
#     iteration = 0

#     # 3) Loop until convergence or max iterations
#     while iteration < max_iterations:
#         iteration += 1
#         print(f'iteration {iteration}')
#         # 3a) Compute shift via phase correlation method        
#         center = phase_correlation_method(JF_Img, center, mask, cropsize= 2048)

#         # 3b) Check for convergence: shifts below 0.5 pixel in both directions
#         if abs(center[0]-centers[-1][0]) < 1 and abs(center[1]-centers[-1][1]) < 1:
#             # Converged: return final center and history
#             return center, centers

#         # 3d) boundary check, the center won't be too far away from the detector center
#         center[0] = min(center[0], init_center_guess[0]+15)
#         center[0] = max(center[0], init_center_guess[0]-15)
#         center[1] = min(center[1], init_center_guess[1]+15)
#         center[1] = max(center[1], init_center_guess[1]-15)
        
#         centers.append(center.copy())

#     # 4) If we exit loop without convergence, warn and return None
#     print('Warning: iterative_center_of_mass did not converge after', max_iterations, 'iterations')
#     return None, centers
    