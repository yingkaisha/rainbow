import sys
import time
import argparse
from glob import glob
from datetime import datetime, timedelta

# data tools
import h5py
import numba as nb
import numpy as np
import multiprocessing

# custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/utils/')

import data_utils as du
from namelist import * 

@nb.njit(fastmath=True)
def nonzero_ind(array):
    for ind, val in enumerate(array):
        if val > 0:
            return ind
    # if all-zero
    print("All zero identidied, return 94-th")
    return 97

@nb.njit(fastmath=True)
def cdf_loss(precip_cdf, i, j, m, n):
    '''
    Precipitation cdf loss. Implemented as in Hamill et al. 2017
    '''
    q_bins = np.array([ 1.0e-04, 5.0e-04, 1.0e-03, 5.0e-03, 1.0e-02, 2.0e-02, 3.0e-02,
                        4.0e-02, 5.0e-02, 6.0e-02, 7.0e-02, 8.0e-02, 9.0e-02, 1.0e-01,
                        1.1e-01, 1.2e-01, 1.3e-01, 1.4e-01, 1.5e-01, 1.6e-01, 1.7e-01,
                        1.8e-01, 1.9e-01, 2.0e-01, 2.1e-01, 2.2e-01, 2.3e-01, 2.4e-01,
                        2.5e-01, 2.6e-01, 2.7e-01, 2.8e-01, 2.9e-01, 3.0e-01, 3.1e-01,
                        3.2e-01, 3.3e-01, 3.4e-01, 3.5e-01, 3.6e-01, 3.7e-01, 3.8e-01,
                        3.9e-01, 4.0e-01, 4.1e-01, 4.2e-01, 4.3e-01, 4.4e-01, 4.5e-01,
                        4.6e-01, 4.7e-01, 4.8e-01, 4.9e-01, 5.0e-01, 5.1e-01, 5.2e-01,
                        5.3e-01, 5.4e-01, 5.5e-01, 5.6e-01, 5.7e-01, 5.8e-01, 5.9e-01,
                        6.0e-01, 6.1e-01, 6.2e-01, 6.3e-01, 6.4e-01, 6.5e-01, 6.6e-01,
                        6.7e-01, 6.8e-01, 6.9e-01, 7.0e-01, 7.1e-01, 7.2e-01, 7.3e-01,
                        7.4e-01, 7.5e-01, 7.6e-01, 7.7e-01, 7.8e-01, 7.9e-01, 8.0e-01,
                        8.1e-01, 8.2e-01, 8.3e-01, 8.4e-01, 8.5e-01, 8.6e-01, 8.7e-01,
                        8.8e-01, 8.9e-01, 9.0e-01, 9.1e-01, 9.2e-01, 9.3e-01, 9.4e-01,
                        9.5e-01])
    
    A_ij = precip_cdf[:, i, j]
    A_mn = precip_cdf[:, m, n]
    pmin = nonzero_ind(A_ij)
    qmin = 100*q_bins[pmin]
    
    return np.sum(np.abs(A_ij-A_mn)[pmin:])/(95-qmin)

@nb.njit(fastmath=True)
def elev_loss(elev_land, i, j, m, n):
    '''
    elev loss. Implemented as in Hamill et al. 2017
    '''
    Z_ij = elev_land[i, j]
    Z_mn = elev_land[m, n]
    return 1-1/np.exp(np.abs(Z_ij-Z_mn)/2500)

@nb.njit(fastmath=True)
def adjacent(x1, x2):
    diffx = np.abs(x1 - x2)
    return np.min(np.array([diffx, np.abs(diffx+8), np.abs(diffx-8)]))

@nb.njit(fastmath=True)
def facet_loss(facet_h, facet_m, facet_l, W_facet, i, j, m, n):
    Fh_ij = facet_h[i, j]
    Fm_ij = facet_m[i, j]
    Fl_ij = facet_l[i, j]
    
    Fh_mn = facet_h[m, n]
    Fm_mn = facet_m[m, n]
    Fl_mn = facet_l[m, n]
    
    Dh = adjacent(Fh_ij, Fh_mn)
    Dm = adjacent(Fm_ij, Fm_mn)
    Dl = adjacent(Fl_ij, Fl_mn)
    
    return W_facet[i, j]*(Dh + Dm + Dl)/3

@nb.njit(fastmath=True)
def distance_loss(i, j, m, n):
    return np.sqrt((i-m)**2 + (j-n)**2)

@nb.njit(fastmath=True)
def loss_cal(precip_cdf, elev_land, facet_h, facet_m, facet_l, W_facet, i, j, m, n):
    return 0.1*cdf_loss(precip_cdf, i, j, m, n) + 0.4*elev_loss(elev_land, i, j, m, n) + \
           0.1*facet_loss(facet_h, facet_m, facet_l, W_facet, i, j, m, n) + 0.001*distance_loss(i, j, m, n)

@nb.njit(fastmath=True)
def SL_search(IND, IND_base, precip_cdf, elev_land, facet_h, facet_m, facet_l, W_facet,):
    
    d_min = 9 # d = sqrt(d**2)
    d_max = 125
    Ns = 40
    
    L_IND = len(IND)
    L_IND_base = len(IND_base)
    grid_shape = W_facet.shape
    
    OUT = np.empty((grid_shape)+(Ns+1, 3,))
    OUT_ij = np.empty((Ns+1, 3,)) # +1 for grid point itself
    
    for ind in range(L_IND):
        i = IND[ind, 0]
        j = IND[ind, 1]
        
        OUT_ij[0, 0] = i
        OUT_ij[0, 1] = j
        OUT_ij[0, 2] = 999
        
        for s in range(1, Ns+1):
            # initialize record
            record = 999
            # loop over supplemental locations
            for ind_base in range(L_IND_base):
                m = IND_base[ind_base, 0]
                n = IND_base[ind_base, 1]
                
                # must be close
                dist = np.sqrt((i-m)**2+(j-n)**2)
                if dist < d_max:
                    # apply an extra distance penality
                    if np.min( (OUT_ij[:s, 0]-m)**2+(OUT_ij[:s, 1]-n)**2 ) >= d_min:
                        temp_loss = loss_cal(precip_cdf, elev_land, facet_h, facet_m, facet_l, W_facet, i, j, m, n)
                        if temp_loss < record:
                            record = temp_loss; 
                            temp_outx = m; 
                            temp_outy = n

            OUT_ij[s, 0] = temp_outx
            OUT_ij[s, 1] = temp_outy
            OUT_ij[s, 2] = record

        OUT[i, j, ...] = OUT_ij
    return OUT

# ---------------------------------- #
# parse user inputs
parser = argparse.ArgumentParser()
# positionals
parser.add_argument('month', help='month')
args = vars(parser.parse_args())
month = int(args['month'])
print("month = {}".format(month))

# ---------------------------------- #
# domain information and elevation
with h5py.File(save_dir+'BC_domain_info.hdf', 'r') as h5io:
    base_lon = h5io['base_lon'][...]
    base_lat = h5io['base_lat'][...]
    etopo_025 = h5io['etopo_base'][...]
    land_mask = h5io['land_mask_base'][...]
    land_mask_bc = h5io['land_mask_bc'][...]

# maskout low quality facet grid points at the edge
land_mask[:5, :] = True
land_mask[:, :5] = True
land_mask[-4:, :] = True
land_mask[:, -4:] = True
bc_in_base = np.ones(land_mask.shape).astype(bool)
bc_in_base[bc_inds[0]:bc_inds[1], bc_inds[2]:bc_inds[3]] = land_mask_bc

Z = np.copy(etopo_025)
Z[land_mask] = np.nan

# ---------------------------------- #
# facet
with h5py.File(save_dir+'NA_SL_info.hdf', 'r') as h5io:
    facet_h = h5io['facet_h'][...]
    facet_m = h5io['facet_m'][...]
    facet_l = h5io['facet_l'][...]
    W_facet = h5io['W_facet'][...]
    
# ---------------------------------- #
# ERA5 quantiles
with h5py.File(ERA_dir+'PT_3hour_q.hdf', 'r') as h5io:
    era_3hq = h5io['era_3hq'][month, ...]
    
era_3hq = 2.0*era_3hq # 3-hr to 6-hr
era_3hq = era_3hq[:99, ...] # subset to 95-th, avoiding higher extremes
era_3hq[..., land_mask] = np.nan

# ---------------------------------- #
grid_shape = land_mask.shape
# subsetting by land mask
IND = []; IND_base = []
for i in range(grid_shape[0]):
    for j in range(grid_shape[1]):
        if ~bc_in_base[i, j]:
            IND.append([i, j])
        if ~land_mask[i, j]:
            IND_base.append([i, j])
IND = np.array(IND, dtype=np.int)
IND_base = np.array(IND_base, dtype=np.int)

# ---------------------------------- #
# the main loop
print("Main program starts ...")
start_time = time.time()
OUT = SL_search(IND, IND_base, era_3hq, Z, facet_h, facet_m, facet_l, W_facet,)
print("{} secs for all locs".format(time.time()-start_time))
# ---------------------------------- #
# save
tuple_save = (OUT,)
label_save = ['IND']
du.save_hdf5(tuple_save, label_save, save_dir, 'S40_mon{}.hdf'.format(month))
