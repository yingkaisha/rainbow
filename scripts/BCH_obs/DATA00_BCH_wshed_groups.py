import sys
from glob import glob

import h5py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/QC_OBS/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/QC_OBS/utils')

import graph_utils as gu
import data_utils as du
import BCH_utils as bu
from namelist import *

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

# ========== Region subsets ========== #

south = [-130, -121, 48.75, 50.25]
north = [-127.5, -110, 53, 60]

loc_id = [] # 0 van isl, 1 south, 2 rocky, 3 north

for i in range(len(stn_code)):
    stn_lat_temp = stn_lat[i]
    stn_lon_temp = stn_lon[i]
    
    if du.check_bounds(stn_lon_temp, stn_lat_temp, south):
        loc_id.append(1)
    elif du.check_bounds(stn_lon_temp, stn_lat_temp, north):
        loc_id.append(3)
    else:
        loc_id.append(2)
        
loc_id = np.array(loc_id)

flag_sw = loc_id == 1
flag_si = loc_id == 2
flag_n = loc_id == 3

# ========== Save ========== #

tuple_save = (stn_lon, stn_lat, flag_sw, flag_si, flag_n)
label_save = ['stn_lon', 'stn_lat', 'flag_sw', 'flag_si', 'flag_n']
du.save_hdf5(tuple_save, label_save, save_dir, 'BCH_wshed_groups.hdf')


