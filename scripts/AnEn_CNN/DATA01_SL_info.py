import sys
import time
import os.path
from glob import glob
from datetime import datetime, timedelta

# data tools
import h5py
import numpy as np
import numba as nb

from scipy.ndimage import gaussian_filter

# custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/utils/')

import data_utils as du
from namelist import * 

def window_stdev_slow(arr, radius=1):
    grid_shape = arr.shape
    out = np.empty(arr.shape)
    out[...] = np.nan
    for i in range(radius, grid_shape[0]-radius+1):
        for j in range(radius, grid_shape[1]-radius+1):
            window = arr[i-radius:i+radius+1, j-radius:j+radius+1].ravel()
            sigma_ij = np.nanstd(window)
            out[i, j] = sigma_ij
    return out

def facet(Z):
    '''
    compass = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    '''
    dZy, dZx = np.gradient(Z)
    dZy = -1*dZy
    dZx = -1*dZx
    Z_to_deg = np.arctan2(dZx, dZy)/np.pi*180
    Z_to_deg[Z_to_deg<0] += 360
    Z_ind = np.round(Z_to_deg/45.0)

    thres = np.sqrt(dZy**2+dZx**2) < 0.1
    Z_ind[thres] = 8
    Z_ind = Z_ind.astype(int)

    return facet_group(Z_ind, rad=1)

def facet_group(compass, rad):
    thres = rad*4
    grid_shape = compass.shape
    compass_pad = np.pad(compass, 2, constant_values=999)
    
    out = np.empty(grid_shape)
    
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            group = compass_pad[i-rad:i+rad+1, j-rad:j+rad+1].ravel()
            flag_clean = ~(group==999)
            if np.sum(flag_clean)<thres:
                out[i, j] = np.nan
            else:
                group_clean = group[flag_clean]
                out[i, j] = Gibbs_rule(group_clean)
    return out
            
def adjacent(x1, x2):
    diffx = np.abs(x1 - x2)
    return np.min(np.array([diffx, np.abs(diffx+8), np.abs(diffx-8)]))

def sum_adjacent(counts, n0):
    n0_left = n0-1
    if n0_left < 0:
        n0_left += 8
    n0_right = n0+1
    if n0_right > 7:
        n0_right -= 8
    return np.max(np.array([counts[n0]+counts[n0_left], counts[n0]+counts[n0_right]])) 

def Gibbs_rule(compass_vec):
    L = len(compass_vec)
    counts = np.bincount(compass_vec, minlength=9)
    count_sort = np.argsort(counts)[::-1]
    
    no0 = count_sort[0]
    no1 = count_sort[1]
    no2 = count_sort[2]
    no3 = count_sort[3]
    
    num_no0 = counts[no0]
    num_no1 = counts[no1]
    num_no2 = counts[no2]
    num_no3 = counts[no3]

    sum_no0 = sum_adjacent(counts, no0)
    sum_no1 = sum_adjacent(counts, no1)
    
    # 1 + 2 > 50%
    if num_no0 + num_no1 > 0.5*L:
        # 1-2 >= 20%, or 1, 2, 3 flat, or 1 adj to 2, 3
        if num_no0-num_no1 >= 0.2*L \
        or no0 == 8 or no1 == 8 or no2 == 8 \
        or adjacent(no0, no1) == 1 or adjacent(no0, no2) == 1:
            return no0
        else:
            # 1 not adj to 2 or 3, and 2 not adj to 3
            if adjacent(no0, no1) > 1 and adjacent(no0, no2) > 1 and adjacent(no1, no2) > 1:
                return no0
            else:
                # 1 adj to 4, 2 adj to 3
                if adjacent(no0, no3) == 1 and adjacent(no1, no2) == 1:
                    if num_no2-num_no3 <= 0.1*L:
                        if num_no0+num_no3 > num_no1+num_no2:
                            return no0
                        else:
                            return no1
                    else:
                        if num_no1 + num_no2 > num_no0:
                            return no1
                        else:
                            return no0
                else:
                    # 2 adj to 3
                    if adjacent(no1, no2) == 1:
                        if num_no1 + num_no2 > num_no0:
                            return no1
                        else:
                            return no0
                    else:
                        # impossible
                        return acdabbfatsh
    else:
        # 1 adj to 2, 1 not flat, 2 not flat
        if adjacent(no0, no1) == 1 and no0 != 8 and no1 != 8:
            return no0
        else:
            # 1 not adj to 2, 1 not flat, 2 not flat
            if no0 != 8 and no1 != 8 and adjacent(no0, no1) > 1:
                if sum_no0 > sum_no1:
                    return no0
                else:
                    return no1
            else:
                if no0 == 8 or no1 == 8:
                    # 1 is flat
                    if no0 == 8:
                        if sum_no1 > num_no0:
                            return no1
                        else:
                            return no0
                    else:
                        if num_no0 >= num_no1:
                            return no0
                        else:
                            return no1
                else:
                    # impossible
                    return afegdagt
                
                
# importing domain information
with h5py.File(save_dir+'BC_domain_info.hdf', 'r') as h5io:
    base_lon = h5io['base_lon'][...]
    base_lat = h5io['base_lat'][...]
    etopo_025 = h5io['etopo_base'][...]
    sigma025 = h5io['sigma025'][...]
    land_mask = h5io['land_mask_base'][...]
    land_mask_bc = h5io['land_mask_bc'][...]

etopo_025[land_mask] = 0

Z_l = gaussian_filter(etopo_025, 10/np.pi)
Z_l[land_mask] = 0
Z_m = gaussian_filter(etopo_025, 5/np.pi)
Z_m[land_mask] = 0
Z_h = np.copy(etopo_025)
Z_h[land_mask] = 0

facet_h = facet(Z_h)
facet_m = facet(Z_m)
facet_l = facet(Z_l)

facet_h[land_mask] = np.nan
facet_m[land_mask] = np.nan
facet_l[land_mask] = np.nan

sigma_facet = window_stdev_slow(etopo_025, radius=5)
W_facet = sigma_facet/np.nanmax(sigma_facet)

# W_025 = 0.6*(sigma025-15)
# W_025[W_025>0.6] = 0.6
# W_025[W_025<0.2] = 0.2
# W_025[land_mask] = np.nan

W_025 = 0.8*(sigma025-15)
W_025[W_025>0.8] = 0.8
W_025[W_025<0.2] = 0.2
W_025[land_mask] = np.nan

tuple_save = (facet_h, facet_m, facet_l, W_facet, W_025)
label_save = ['facet_h', 'facet_m', 'facet_l', 'W_facet', 'W_SL']
du.save_hdf5(tuple_save, label_save, save_dir, 'NA_SL_info.hdf')
