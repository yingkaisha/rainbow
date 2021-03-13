'''
Climatological mean
'''

import sys
from glob import glob

import h5py
import numpy as np
import numba as nb

import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/QC_OBS/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/QC_OBS/utils')

import graph_utils as gu
import data_utils as du
import BCH_utils as bu
from namelist import *

@nb.njit()
def climo_mean(fcst, flag_pick):
    
    N_grids = fcst.shape[-1]
    
    MEAN = np.empty((12, N_grids))
    flag_pick = flag_pick > 0
    
    for mon in range(12):
        flag_pick_temp = flag_pick[mon, :]
        cnn_sub = fcst[flag_pick_temp, ...]
        
        for n in range(N_grids):
            temp = cnn_sub[..., n].ravel()
            flag_nan = np.logical_not(np.isnan(temp))
            MEAN[mon, n] = np.mean(temp[flag_nan])
    return MEAN


# importing domain information
with h5py.File(save_dir+'BC_domain_info.hdf', 'r') as h5io:
    bc_lon = h5io['bc_lon'][...]
    bc_lat = h5io['bc_lat'][...]

with pd.HDFStore(BACKUP_dir+'BCH_85_metadata.hdf', 'r') as hdf_temp:
    metadata = hdf_temp['metadata']

stn_code = metadata['code'].values
stn_lat = metadata['lat'].values
stn_lon = metadata['lon'].values

with pd.HDFStore(BACKUP_dir+'BCH_PREC_QC_3H_2016_2020.hdf', 'r') as hdf_io:
    keys = hdf_io.keys()
keys = du.del_slash(keys)

flag_pick = []
for key in stn_code:
    if key in keys:
        flag_pick.append(True)
    else:
        flag_pick.append(False)
        
flag_pick = np.array(flag_pick)
stn_code = stn_code[flag_pick]
stn_lat = stn_lat[flag_pick]
stn_lon = stn_lon[flag_pick]

# ========== params ========== #

indx, indy = du.grid_search(bc_lon, bc_lat, stn_lon, stn_lat)

# ========== Collecting ERA5 P_grid ========== # 

ERA5 = ()
for year in range(2000, 2015):
    print('Importing ERA5 {}'.format(year))
    if year%4 == 0:
        N_days = 366
    else:
        N_days = 365
    
    with h5py.File(ERA_dir+'ERA5_GEFS-fcst_{}.hdf'.format(year), 'r') as h5io:
        era_pct = h5io['era_fcst'][..., bc_inds[0]:bc_inds[1], bc_inds[2]:bc_inds[3]]
    
    ERA5 += (era_pct[..., indx, indy],)

ERA5_obs = np.concatenate(ERA5, axis=0)

# ========== temproal separations ========== # 

N_days = len(ERA5_obs)
date_base = datetime(2000, 1, 1)
date_list = [date_base + timedelta(days=x) for x in np.arange(N_days, dtype=np.float)]

N_fcst = 54
freq = 3.0
FCSTs = np.arange(9, 240+freq, freq) # fcst lead as hour
FCSTs = FCSTs[:54]

N_grids = ERA5_obs.shape[-1]

L = len(date_list)
flag_pick = np.zeros((12, N_fcst, L,))

for mon in range(12):
    # ----- #
    if mon == 0:
        month_around = [11, 0, 1] # handle Jan
    elif mon == 11:
        month_around = [10, 11, 0] # handle Dec
    else:
        month_around = [mon-1, mon, mon+1]
    
    for lead in range(N_fcst):
        for d, date in enumerate(date_list):
            # ini date + lead time
            date_true = date + timedelta(hours=FCSTs[lead])

            if date_true.month-1 in month_around: 
                flag_pick[mon, lead, d] = 1.0
            else:
                flag_pick[mon, lead, d] = 0.0

MEAN = np.empty((12, N_fcst, N_grids))

for lead in range(N_fcst):
    MEAN[:, lead, :] = climo_mean(ERA5_obs[:, lead, :], flag_pick[:, lead, :])

# ---------- Duplicate to 2016-2020 ---------- #

N_days = 366 + 365*3
date_base = datetime(2016, 1, 1)
date_list = [date_base + timedelta(days=x) for x in np.arange(N_days, dtype=np.float)]

MEAN_BCH = np.empty((N_days, N_fcst, N_grids))

for i, date in enumerate(date_list):
    mon_ = date.month-1
    MEAN_BCH[i, :, :] = MEAN[mon, ...]
    
tuple_save = (MEAN_BCH,)
label_save = ['MEAN_BCH']
du.save_hdf5(tuple_save, label_save, save_dir, 'BCH_climo-mean_ERA5.hdf')
