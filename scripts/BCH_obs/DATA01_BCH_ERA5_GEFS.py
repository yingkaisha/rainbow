'''
Creating paired, ERA5, GEFS, BC Hydro 3 hourly precipitation.
'''


import sys
from glob import glob

import h5py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/QC_OBS/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/QC_OBS/utils')

import data_utils as du
import BCH_utils as bu
from namelist import *

# ========== station metadata (84 stations) ========== #

with h5py.File(save_dir+'BC_domain_info.hdf', 'r') as h5io:
    base_lon = h5io['base_lon'][...]
    base_lat = h5io['base_lat'][...]
    bc_lon = h5io['bc_lon'][...]
    bc_lat = h5io['bc_lat'][...]
    etopo_bc = h5io['etopo_bc'][...]
    land_mask = h5io['land_mask_base'][...]
    land_mask_bc = h5io['land_mask_bc'][...]
    
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

N_fcst = 54
indx, indy = du.grid_search(bc_lon, bc_lat, stn_lon, stn_lat)

# ========== ERA5 ========== #

ERA5 = ()
for year in range(2016, 2020):
    with h5py.File(ERA_dir+'ERA5_GEFS-fcst_{}.hdf'.format(year), 'r') as h5io:
        era_ = h5io['era_fcst'][..., :N_fcst, bc_inds[0]:bc_inds[1], bc_inds[2]:bc_inds[3]]
    ERA5 += (era_[..., indx, indy],)
    
ERA5_obs = np.concatenate(ERA5, axis=0)

# ========== GEFS ========== #

GEFS = ()
for year in range(2016, 2020):
    with h5py.File(REFCST_dir+'En_mean_APCP_{}.hdf'.format(year), 'r') as h5io:
        gefs_ = h5io['bc_mean'][:, :N_fcst, ...]
    GEFS += (gefs_[..., indx, indy],)

GEFS_obs = np.concatenate(GEFS, axis=0)

# ========== BCH ========== #

OBS = ()
for key in stn_code:
    with pd.HDFStore(BACKUP_dir+'BCH_PREC_QC_3H_2016_2020.hdf', 'r') as hdf_io:
        pd_temp = hdf_io[key]
        pd_temp.index = pd_temp['datetime']
        pd_temp = pd_temp['2016-01-01':'2021-01-01']
        obs_ = pd_temp['PREC_HOUR_QC'].values
    OBS += (obs_[:, None],)

BCH_obs = np.concatenate(OBS, axis=-1)

N_days = 366 + 365*3
date_base = datetime(2016, 1, 1)
date_list = [date_base + timedelta(days=x) for x in np.arange(N_days, dtype=np.float)]

FCSTs = np.arange(9.0, 24*9+3, 3)
FCSTs = FCSTs[:N_fcst]

BCH_lead = np.empty((N_days, N_fcst, len(stn_code)))

# map obs to ini and lead times
for day, date in enumerate(date_list):
    for t, fcst_temp in enumerate(FCSTs):
        
        # fcsted (targeted) date
        date_true = date + timedelta(hours=fcst_temp)
        ind_true = int((date_true-date_base).total_seconds()/60/60/3.0)
        
        BCH_lead[day, t, :] = BCH_obs[ind_true, :]


BCH_lead[BCH_lead>100] = np.nan 
        
# ========== Save ========== #

tuple_save = (ERA5_obs, GEFS_obs, BCH_lead, flag_pick, indx, indy)
label_save = ['ERA5_obs', 'GEFS_obs', 'BCH_obs', 'stn_flag', 'indx', 'indy']
du.save_hdf5(tuple_save, label_save, save_dir, 'BCH_ERA5_GEFS_pairs.hdf')


