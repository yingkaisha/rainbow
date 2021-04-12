
import sys
from glob import glob
from datetime import datetime, timedelta

# data tools
import h5py
import numpy as np
import numba as nb

# custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/utils/')

import data_utils as du
from namelist import * 

@nb.njit()
def q_3h(era_3h, mon_inds):
    q_out = np.empty((12, 107, 160, 220))
    for mon in range(12):
        flag_pick = mon_inds[mon, :]
        for i in range(160):
            for j in range(220):
                q_out[mon, :, i, j] = np.quantile(era_3h[flag_pick, i, j], q_bins)
    return q_out

def moving_accum(data, window):
    N_days, Nx, Ny = data.shape
    N_out = N_days - window + 1
    out = np.empty((N_out, Nx, Ny))
    
    for i in range(N_out):
        temp_ = data[i:i+window, ...]
        out[i, ...] = np.nansum(temp_, axis=0)
    return out

with h5py.File(save_dir+'BC_domain_info.hdf', 'r') as h5io:
    base_lon = h5io['base_lon'][...]
    base_lat = h5io['base_lat'][...]
    lon_025 = h5io['bc_lon'][...]
    lat_025 = h5io['bc_lat'][...]
    land_mask_base = h5io['land_mask_base'][...]
    land_mask_bc = h5io['land_mask_bc'][...]
    
bc_in_base = np.ones(land_mask.shape).astype(bool)
bc_in_base[bc_inds[0]:bc_inds[1], bc_inds[2]:bc_inds[3]] = land_mask_bc
    
base = datetime(2000, 1, 1)
date_list = [base + timedelta(hours=x) for x in range(0, 6940*24, 3)]

mon_inds = np.zeros((12, len(date_list)), dtype=bool)
for mon in range(12):
    if mon == 0:
        month_around = [11, 0, 1] # handle Jan
    elif mon == 11:
        month_around = [10, 11, 0] # handle Dec
    else:
        month_around = [mon-1, mon, mon+1]
    for i, date in enumerate(date_list):
        if date.month-1 in month_around:
            mon_inds[mon, i] = True

q_bins = np.concatenate((np.array([0.0001, 0.0005, 0.001, 0.005]), 
                         np.arange(0.01, 1, 0.01), 
                         np.array([0.995, 0.999, 0.9995, 0.9999])), axis=0)

with h5py.File(ERA_dir+'PT_3hour_combine.hdf', 'r') as h5io:
    era_3h = h5io['era_3h'][...]
    
q_out = q_3h(era_3h, mon_inds)

era_3h_7day = moving_accum(era_3h, N_fcst)
L = len(era_3h_7day)

q_out_accum = q_3h(era_3h_7day, mon_inds[:, :L])

q_out_bc = q_out[..., ~bc_in_base]
q_out_accum_bc = q_out_accum[..., ~bc_in_base]

tuple_save = (q_out, q_out_bc, q_out_accum, q_out_accum_bc, q_bins)
label_save = ['era_3hq_base', 'era_3hq_bc', 'era_3hq_accum_base', 'era_3hq_accum_bc', 'q_bins']
du.save_hdf5(tuple_save, label_save, ERA_dir, 'PT_3hour_q.hdf')
