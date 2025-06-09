import numpy as np
from scipy.optimize import curve_fit
import copy
from scipy.optimize import OptimizeWarning
import warnings


tile_start_y=[0, 149, 550, 699, 1100, 1249, 1649, 1798]
tile_start_x=[149, 1185, 149, 1185, 0, 1037, 0, 1037]
size_x = 1024
size_y = 512

# Function to model the Gaussian
def gaussian(x, a, b, c):
    return a * np.exp(-(x - b)**2 / (2 * c**2))

def ComModCor_1area(Area):
    data = Area.ravel()
    # Generate histogram
    counts, bin_edges = np.histogram(data, bins=100, range=(-1, 1))

    # Bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find the bin that contains the value 0 and exclude it (because on the histogram there will be a stick structure)
    zero_bin_index = np.where((bin_edges[:-1] <= 0) & (bin_edges[1:] >= 0))[0][0]
    filtered_counts = np.delete(counts, zero_bin_index+1)[:65] # [:65] means we only use histogram within the range [-1, 0.3] to fit the gaussian curve
    filtered_bin_centers = np.delete(bin_centers, zero_bin_index+1)[:65]

    # Fit the Gaussian to the data
    # only with those pixels with pixel value within the range of (-1, 0.5)

    try:
        # Perform curve fitting
        popt, pcov = curve_fit(gaussian, filtered_bin_centers, filtered_counts, p0=[14000, 0, 0.25], maxfev=800)
    except RuntimeError as e:
        print(f"Curve fitting failed: {e}")
        # Optionally, you can set popt to some default values or None if the fit fails
        popt = [0,0,0]
    except OptimizeWarning as w:
        popt = [0,0,0]
        print(f"Optimization warning: {w}")

    Avg = popt[1]
    Var = popt[2]

    # smartly choose the threshold
    start_x = Avg + Var
    end_x = Avg + 3*Var
    bin_center_mask = (bin_centers >= start_x) & (bin_centers <= end_x)
    alpha = 0
    for bin_center in bin_centers[bin_center_mask]:
        hist_val = counts[bin_centers == bin_center]
        fitting_val = gaussian(bin_center, popt[0], popt[1], popt[2])
        deviation = np.abs(hist_val - fitting_val)/fitting_val
        if deviation > 0.1:
            alpha = (bin_center-Avg)/Var
            # print(f'No.{i} alpha = {alpha:.3g}')
            break
    Area_copy = copy.deepcopy(Area)
    Area_copy[Area_copy < Avg + alpha*Var] = 0 # cut to zero
    Area_copy[Area_copy >= Avg + alpha*Var] -= Avg # offset shift
    return Area_copy, popt, [counts, bin_edges]

# This is tile by tile
# def ComModCor_single(JFImg, tile_start_x = tile_start_x, tile_start_y = tile_start_y, size_x = size_x, size_y = size_y):

#     New_Img = copy.deepcopy(JFImg)
#     Optimal_Parameters, Histograms = [], []
#     mask = np.zeros_like(JFImg)

#     for i in range(len(tile_start_y)):

#         data = JFImg[tile_start_y[i]:tile_start_y[i]+size_y, tile_start_x[i]:tile_start_x[i]+size_x].ravel()
#         # Generate histogram
#         counts, bin_edges = np.histogram(data, bins=100, range=(-1, 1))
#         Histograms.append([counts, bin_edges])

#         # Bin centers
#         bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

#         # Find the bin that contains the value 0 and exclude it (because on the histogram there will be a stick structure)
#         zero_bin_index = np.where((bin_edges[:-1] <= 0) & (bin_edges[1:] >= 0))[0][0]
#         filtered_counts = np.delete(counts, zero_bin_index+1)[:65] # [:65] means we only use histogram within the range [-1, 0.3] to fit the gaussian curve
#         filtered_bin_centers = np.delete(bin_centers, zero_bin_index+1)[:65]

#         # Fit the Gaussian to the data
#         # only with those pixels with pixel value within the range of (-1, 0.5)
#         popt, pcov = curve_fit(gaussian, filtered_bin_centers, filtered_counts, p0=[14000, 0, 0.25])

#         Optimal_Parameters.append(popt)
#         Avg = popt[1]
#         Var = popt[2]
#         Cropped_Img = New_Img[tile_start_y[i]:tile_start_y[i]+size_y, tile_start_x[i]:tile_start_x[i]+size_x]
#         Cropped_Img[Cropped_Img < Avg + 1.5*Var] = 0 # cut to zero
#         Cropped_Img[Cropped_Img >= Avg + 1.5*Var] -= Avg # offset shift
#         mask[tile_start_y[i]:tile_start_y[i]+size_y, tile_start_x[i]:tile_start_x[i]+size_x] = 1


#     # The rest of pixels should be zero
#     New_Img[mask == 0] = 0

#     return New_Img, Optimal_Parameters, Histograms

# This is column by column
def ComModCor_single(JFImg, tile_start_x = tile_start_x, tile_start_y = tile_start_y, size_x = size_x, size_y = size_y):
    # This is updated, and I want the common mode correction column by column
    # One tile can be separated into 4 columns along the x direction
    tile_start_x = [tile_start_x[i] + 256 * j for j in range(4) for i in range(8)]
    tile_start_y = [tile_start_y[i] + 0 * j for j in range(4) for i in range(8)]
    size_x = size_x//4
    New_Img = copy.deepcopy(JFImg)
    Optimal_Parameters, Histograms = [], []
    mask = np.zeros_like(JFImg)
    
    for i in range(len(tile_start_y)):

        data = JFImg[tile_start_y[i]:tile_start_y[i]+size_y, tile_start_x[i]:tile_start_x[i]+size_x].ravel()
        # Generate histogram
        counts, bin_edges = np.histogram(data, bins=100, range=(-1, 1))
        Histograms.append([counts, bin_edges])

        # Bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Find the bin that contains the value 0 and exclude it (because on the histogram there will be a stick structure)
        zero_bin_index = np.where((bin_edges[:-1] <= 0) & (bin_edges[1:] >= 0))[0][0]
        filtered_counts = np.delete(counts, zero_bin_index+1)[:65] # [:65] means we only use histogram within the range [-1, 0.3] to fit the gaussian curve
        filtered_bin_centers = np.delete(bin_centers, zero_bin_index+1)[:65]

        # Fit the Gaussian to the data
        # only with those pixels with pixel value within the range of (-1, 0.5)

        try:
            # Perform curve fitting
            popt, pcov = curve_fit(gaussian, filtered_bin_centers, filtered_counts, p0=[14000, 0, 0.25], maxfev=800)
        except RuntimeError as e:
            print(f"Curve fitting failed: {e}")
            # Optionally, you can set popt to some default values or None if the fit fails
            popt = [0,0,0]
        except OptimizeWarning as w:
            popt = [0,0,0]
            print(f"Optimization warning: {w}")
        Optimal_Parameters.append(popt)
        Avg = popt[1]
        Var = popt[2]
        Cropped_Img = New_Img[tile_start_y[i]:tile_start_y[i]+size_y, tile_start_x[i]:tile_start_x[i]+size_x]
        # smartly choose the threshold
        start_x = Avg + Var
        end_x = Avg + 3*Var
        bin_center_mask = (bin_centers >= start_x) & (bin_centers <= end_x)
        alpha = 0
        for bin_center in bin_centers[bin_center_mask]:
            hist_val = counts[bin_centers == bin_center]
            fitting_val = gaussian(bin_center, popt[0], popt[1], popt[2])
            deviation = np.abs(hist_val - fitting_val)/fitting_val
            if deviation > 0.1:
                alpha = (bin_center-Avg)/Var
                # print(f'No.{i} alpha = {alpha:.3g}')
                break
        Cropped_Img[Cropped_Img < Avg + alpha*Var] = 0 # cut to zero
        Cropped_Img[Cropped_Img >= Avg + alpha*Var] -= Avg # offset shift
        mask[tile_start_y[i]:tile_start_y[i]+size_y, tile_start_x[i]:tile_start_x[i]+size_x] = 1

    
    # The rest of pixels should be zero
    New_Img[mask == 0] = 0

    return New_Img, Optimal_Parameters, Histograms


# def ComModCor_single(JFImg, tile_start_x = tile_start_x, tile_start_y = tile_start_y, size_x = size_x, size_y = size_y):
#     # This is updated, and I want the common mode correction column by column
#     # One tile can be separated into 4 columns along the x direction
#     tile_start_x = [tile_start_x[i] + 256 * j for j in range(4) for i in range(8)]
#     tile_start_y = [tile_start_y[i] + 0 * j for j in range(4) for i in range(8)]
#     size_x = size_x//4
#     New_Img = copy.deepcopy(JFImg)
#     Optimal_Parameters, Histograms = [], []
#     mask = np.zeros_like(JFImg)

#     for i in range(len(tile_start_y)):

#         Area_i = New_Img[tile_start_y[i]:tile_start_y[i]+size_y, tile_start_x[i]:tile_start_x[i]+size_x]

#         Cropped_Img[Cropped_Img < Avg + alpha*Var] = 0 # cut to zero
#         Cropped_Img[Cropped_Img >= Avg + alpha*Var] -= Avg # offset shift
#         mask[tile_start_y[i]:tile_start_y[i]+size_y, tile_start_x[i]:tile_start_x[i]+size_x] = 1


#     # The rest of pixels should be zero
#     New_Img[mask == 0] = 0

#     return New_Img, Optimal_Parameters, Histograms


def ComModCor(JFImgs, tile_start_x = tile_start_x, tile_start_y = tile_start_y, size_x = size_x, size_y = size_y):
    if JFImgs.ndim == 2:
        return ComModCor_single(JFImgs, tile_start_x, tile_start_y, size_x, size_y)
    else:
        New_Imgs, Opt_Paras, Hists = [], [], []
        for JFImg in JFImgs:
            New_Img, Optimal_Parameters, Histograms = ComModCor_single(JFImg, tile_start_x = tile_start_x, tile_start_y = tile_start_y, size_x = size_x, size_y = size_y)
            New_Imgs.append(New_Img) # shaoe = (num_of_Imgs, nline, ncolumn)
            Opt_Paras.append(Optimal_Parameters) # shape = (num_of_Imgs, 8, 3), 3 means 3 parameters for each gaussian curve, 8 means 8 tiles
            Hists.append(Histograms) # shape = (num_of_Imgs, 8, 2), 2 including counts and bin_edges
        New_Imgs = np.array(New_Imgs)
        return New_Imgs, Opt_Paras, Hists

