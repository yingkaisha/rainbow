'''
Creating paired, daily ERA5, BC Hydro precipitation values on station locations.
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

indx, indy = du.grid_search(bc_lon, bc_lat, stn_lon, stn_lat)

# ========== ERA5 ========== #

ERA5 = ()
for year in range(2016, 2020):
    if year%4 == 0:
        N_days = 366
    else:
        N_days = 365
    
    with h5py.File(ERA_dir+'PT_3hr_{}.hdf'.format(year), 'r') as h5io:
        era_pct = h5io['era_025'][..., bc_inds[0]:bc_inds[1], bc_inds[2]:bc_inds[3]]
        
    era_pct_daily = np.sum(np.reshape(era_pct, (N_days, 8, 48, 112)), axis=1)
    ERA5 += (era_pct_daily[..., indx, indy],)

ERA5_obs = np.concatenate(ERA5, axis=0)
    
# ========== BCH ========== #

OBS = ()
for key in stn_code:
    with pd.HDFStore(BACKUP_dir+'BCH_PREC_QC_1D_2016_2020.hdf', 'r') as hdf_io:
        pd_temp = hdf_io[key]
        pd_temp.index = pd_temp['datetime']
        pd_temp = pd_temp['2016-01-01':'2019-12-31']
        obs_ = pd_temp['PREC_HOUR_QC'].values
    OBS += (obs_[:, None],)

BCH_obs = np.concatenate(OBS, axis=-1)

# ! <--- data cleaning 
BCH_obs[BCH_obs>100] = np.nan

tuple_save = (ERA5_obs, BCH_obs, flag_pick)
label_save = ['ERA5_obs', 'BCH_obs', 'stn_flag']
du.save_hdf5(tuple_save, label_save, save_dir, 'BCH_ERA5_1D_pairs.hdf')
