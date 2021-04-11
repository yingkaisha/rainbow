'''
Computing the CRPS and MAE of BCH obs and AnEn-SG, AnEn-CNN outputs. 
'''

import sys
import argparse
from glob import glob
from datetime import datetime, timedelta

# data tools
import h5py
import time
import numpy as np

# custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/')
sys.path.insert(0, '/glade/u/home/ksha/PUBLISH/fcstpp/')

from fcstpp import metrics
import analog_utils as ana
import data_utils as du
from namelist import * 

import warnings
warnings.filterwarnings("ignore")

# ---------- Parsers ---------- #
parser = argparse.ArgumentParser()
parser.add_argument('out', help='out')
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

type_ind = int(args['out'])
year = int(args['year'])

if type_ind == 0:
    prefix_raw = 'BASE_final_SS'
    prefix_out = 'BASE_final'
    key_raw = 'AnEn'
    EN = 25
    
elif type_ind == 1:
    prefix_raw = 'SL_final_SS'
    prefix_out = 'SL_final'
    key_raw = 'AnEn'
    EN = 25
    
elif type_ind == 2:
    prefix_raw = 'BASE_CNN'
    prefix_out = 'BASE_CNN'
    key_raw = 'cnn_pred'
    EN = 75 # 25 members dressed to 75
    
elif type_ind == 3:
    prefix_raw = 'SL_CNN'
    prefix_out = 'SL_CNN'
    key_raw = 'cnn_pred'
    EN = 75 # 25 members dressed to 75

# N_days
if year%4 == 0:
    N_days = 366
else:
    N_days = 365

# ========== BCH obs preprocessing ========== # 

# import station obsevations and grid point indices
with h5py.File(save_dir+'BCH_ERA5_3H_verif.hdf', 'r') as h5io:
    BCH_obs = h5io['BCH_obs'][...]
    indx = h5io['indx'][...]
    indy = h5io['indy'][...]
    
# subsetting BCH obs into a given year
N_days_bch = 366 + 365*3
date_base_bch = datetime(2016, 1, 1)
date_list_bch = [date_base_bch + timedelta(days=x) for x in np.arange(N_days_bch, dtype=np.float)]

flag_pick = []
for date in date_list_bch:
    if date.year == year:
        flag_pick.append(True)
    else:
        flag_pick.append(False)

flag_pick = np.array(flag_pick)
    
BCH_obs = BCH_obs[flag_pick, ...]

N_grids = BCH_obs.shape[-1]

# =============== Allocation and MAE computation =============== #

# allocation
## indivual member-based MAE (not the MAE of ensemble mean)
MAE = np.empty((N_days, N_fcst, N_grids)) 
CRPS = np.empty((N_days, N_fcst, N_grids))

# 2d allocations for id 0 and 1
if type_ind == 0 or type_ind == 1:
    with h5py.File(save_dir+'BC_domain_info.hdf', 'r') as h5io:
        land_mask_bc = h5io['land_mask_bc'][...]
    grid_shape = land_mask_bc.shape
    
    AnEn_full = np.empty((365, EN)+grid_shape)
    AnEn_full[...] = np.nan

for lead in range(N_fcst):
    print("computing lead: {}".format(lead))
    
    with h5py.File(REFCST_dir + "{}_{}_lead{}.hdf".format(prefix_raw, year, lead), 'r') as h5io:
        AnEn_ = h5io[key_raw][:, :EN, ...]
    
    if type_ind == 0 or type_ind == 1:
        # id 0 and 1 are flattened grid points, reshape them to 2d.
        AnEn_full[..., ~land_mask_bc] = AnEn_
        AnEn_stn = AnEn_full[..., indx, indy]
    else:
        AnEn_stn = AnEn_[..., indx, indy]
        # cnn outputs can be negative, fix it here.
        AnEn_stn = ana.cnn_precip_fix(AnEn_stn)
        
    crps, mae, _ = metrics.CRPS_1d_nan(BCH_obs[:, lead, :], AnEn_stn)
    MAE[:, lead, ...] = mae
    CRPS[:, lead, ...] = crps

# save (all lead times, per year, per experiment)
tuple_save = (MAE, CRPS,)
label_save = ['MAE', 'CRPS',]
du.save_hdf5(tuple_save, label_save, save_dir, '{}_CRPS_BCH_{}.hdf'.format(prefix_out, year))

