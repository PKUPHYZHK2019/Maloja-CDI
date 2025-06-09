import numpy as np
from scipy.ndimage import correlate


def neighbor_mean(image, mask, kernel_size = 5):

    center = kernel_size//2
    # Define a by default 5x5 kernel with the center set to 0 (exclude the pixel itself)
    kernel = np.ones((kernel_size, kernel_size))
    kernel[center, center] = 0  # Remove the center pixel from the kernel
    
    # Apply the mask (set unwanted pixels to 0 in the image)
    masked_image = np.where(mask == 1, 0, image)
    
    # Apply convolution to sum the neighboring pixels
    neighbor_sum = correlate(masked_image, kernel)
    
    # Count valid neighbors by applying the same kernel to the mask
    # This gives the number of valid (non-masked) neighboring pixels
    valid_neighbors = correlate(1 - mask, kernel)
    
    # Avoid division by zero: set places with no valid neighbors to 1
    valid_neighbors = np.where(valid_neighbors == 0, 1, valid_neighbors)
    
    # Calculate the average of valid neighboring pixels
    average_image = neighbor_sum / valid_neighbors

    return average_image

def abnormal_pixels(image, mask, kernel_size = 5, Z_score_threshold = 3, Value_threshold = 50):
    
    average_img = neighbor_mean(image, mask, kernel_size)
    bad_pix_mask = (image>Z_score_threshold*average_img) & (image > Value_threshold)
    return bad_pix_mask

