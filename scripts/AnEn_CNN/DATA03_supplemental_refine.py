import sys
import time
import os.path
from glob import glob
from datetime import datetime, timedelta

# data tools
import h5py
import numpy as np

# custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/utils/')

import data_utils as du
from namelist import * 

# importing domain information
with h5py.File(save_dir+'BC_domain_info.hdf', 'r') as h5io:
    base_lon = h5io['base_lon'][...]
    base_lat = h5io['base_lat'][...]
    etopo_025 = h5io['etopo_base'][...]
    land_mask = h5io['land_mask_base'][...]
    land_mask_bc = h5io['land_mask_bc'][...]
bc_in_base = np.ones(land_mask.shape).astype(bool)
bc_in_base[bc_inds[0]:bc_inds[1], bc_inds[2]:bc_inds[3]] = land_mask_bc

grid_shape = land_mask.shape
# subsetting by land mask
IND = []
for i in range(grid_shape[0]):
    for j in range(grid_shape[1]):
        if ~bc_in_base[i, j]:
            IND.append([i, j])
IND = np.array(IND, dtype=np.int)
N_grids = len(IND)

# -------------------------------------- #
# Combine and save
N_S = 41
SL_xy = np.zeros((12,)+grid_shape+(N_S, 3,))*np.nan
for month in range(12):
    with h5py.File(save_dir+'S40_mon{}.hdf'.format(month), 'r') as h5io:
        sl_ind = h5io['IND'][...]
    # place the current month into allocation
    SL_xy[month, ...] = sl_ind
        
tuple_save = (SL_xy,)
label_save = ['SL_xy']
du.save_hdf5(tuple_save, label_save, save_dir, 'SL40_d4.hdf')

# -------------------------------------- #
# remove duplicates
N_range = np.array([20, 40])

for N_S in N_range:
    
    with h5py.File(save_dir+'SL40_d4.hdf', 'r') as h5io:
        SL_xy = h5io['SL_xy'][..., :N_S, :]

    inds_to_inds = {}
    # flattening for preserving unique (ix, iy) pairs
    SL_xy_mask = SL_xy[:, ~bc_in_base, :, :]
    Ix = SL_xy_mask[..., 0]
    Iy = SL_xy_mask[..., 1]

    # get unique pairs
    Ix_flat = Ix.reshape(12*N_grids*N_S)
    Iy_flat = Iy.reshape(12*N_grids*N_S)
    IxIy = np.concatenate((Ix_flat[:, None], Iy_flat[:, None]), axis=1)
    IxIy_unique = np.unique(IxIy, axis=0)
    # indx encoding for np.searchsorted
    IxIy_1d = np.sort(IxIy_unique[:, 0]*9.99+IxIy_unique[:, 1]*0.01)

    # map each pair to the unqiue pairs
    for mon in range(12):
        ind_to_ind = np.empty((N_grids, N_S), dtype=np.int)
        for i in range(N_grids):
            ix = Ix[mon, i, :]
            iy = Iy[mon, i, :]

            # applying the same encoding rule
            ixiy_1d = ix*9.99+iy*0.01

            # reverse select inds
            for s in range(N_S):
                ind_to_ind[i, s] = (np.searchsorted(IxIy_1d, ixiy_1d[s]))

        inds_to_inds['{}'.format(mon)] = ind_to_ind
    # applying int
    IxIy_unique = IxIy_unique.astype(np.int)

    # verifing the reverse mapping of inds
    for mon in range(12):
        for i in range(N_grids):
            for s in range(N_S):
                ix = Ix[mon, i, s]
                iy = Iy[mon, i, s]

                ind_to_ind = inds_to_inds['{}'.format(mon)]
                ix_mapped, iy_mapped = IxIy_unique[int(ind_to_ind[i, s]), :]
                # if not matched, then raise a msg
                if (np.abs(ix - ix_mapped) + np.abs(iy - iy_mapped)) > 0:
                    print("no...........")
                    errorerrorerrorerror

    IxIy_maps = tuple(inds_to_inds.values())
    tuple_save = IxIy_maps + (IxIy_unique,)

    label_save = []
    for i in range(12):
        label_save.append('mon_{}_inds'.format(i))
    label_save.append('unique_inds')

    # save
    du.save_hdf5(tuple_save, label_save, save_dir, 'SL{}_d4_unique.hdf'.format(N_S))