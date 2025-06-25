import numpy as np

def is_hit(img, bkg, threshold = 1, percentage = 0.01):

    img_minus_bkg = np.array(img - bkg)
    # number of pixels beyond the threshold
    num_pix_th = np.sum(img_minus_bkg > threshold)
    # total number of pixels
    num_tot = img_minus_bkg.shape[0]*img_minus_bkg.shape[1]

    return num_pix_th/num_tot > percentage


# input a lot of data to decide if they are hits, or backgrounds, or neither
# unit of photonE is keV
def is_hits(imgs, PhotonE = 1, hit_percentage = 0.01, bkg_percentage = 0.002):

    num_pix_1ps = []
    for img in imgs:
        # number of pixels with at least 1 photon
        num_pix_1p = np.sum(img > PhotonE)
        num_pix_1ps.append(num_pix_1p)

    sorted_indices = np.argsort(num_pix_1ps)
    
    # We have to be certain that the last 10% must be background, the hit-rate cannot be that high (90%)
    num_elements_to_take = int(len(num_pix_1ps) * 0.10)
    must_bkg_indices = sorted_indices[:num_elements_to_take]
    must_bkgs = imgs[must_bkg_indices]
    temp_bkg = must_bkgs.mean(0)

    is_hits = []
    is_bkgs = []
    
    for img in imgs:

        # Notice that the bkg_percentage is different from hit_percentage, there will be some data which are
        # neither hits or backgrounds, we just abandon them.

        is_hits.append(is_hit(img, temp_bkg, PhotonE, hit_percentage))
        is_bkgs.append(~is_hit(img, temp_bkg, PhotonE, bkg_percentage))

    return is_hits, is_bkgs

    
        