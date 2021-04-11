'''
Computing the number of members to CRPS relationships of the quantile mapped GEFS based on the BCH observations.
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
import data_utils as du
from namelist import * 


# ---------- Parsers ---------- #
parser = argparse.ArgumentParser()
parser.add_argument('out', help='out')
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

type_ind = int(args['out'])
year = int(args['year'])

if type_ind == 0:
    perfix_smooth = 'BASE_final'
    perfix_raw = 'BASE_final'
    key_smooth = 'AnEn_SG'
    key_raw = 'AnEn'
    
elif type_ind == 1:
    perfix_smooth = 'SL_final'
    perfix_raw = 'SL_final'
    key_smooth = 'AnEn_SG'
    key_raw = 'AnEn'
    
elif type_ind == 2:
    perfix_smooth = 'BASE_CNN'
    perfix_raw = 'BASE_final'
    key_smooth = 'cnn_pred'
    key_raw = 'AnEn'
    
elif type_ind == 3:
    perfix_smooth = 'SL_CNN'
    perfix_raw = 'SL_final'
    key_smooth = 'cnn_pred'
    key_raw = 'AnEn'

print("perfix_smooth = {}; perfix_raw = {}".format(perfix_smooth, perfix_raw))

# ========== BCH obs preprocessing ========== # 

# import station obsevations and grid point indices
with h5py.File(save_dir+'BCH_ERA5_3H_verif.hdf', 'r') as h5io:
    BCH_obs = h5io['BCH_obs'][...]
    indx = h5io['indx'][...]
    indy = h5io['indy'][...]
    
# subsetting BCH obs into a given year
N_days = 366 + 365*3
date_base = datetime(2016, 1, 1)
date_list = [date_base + timedelta(days=x) for x in np.arange(N_days, dtype=np.float)]

flag_pick = []
for date in date_list:
    if date.year == year:
        flag_pick.append(True)
    else:
        flag_pick.append(False)

flag_pick = np.array(flag_pick)
    
BCH_obs = BCH_obs[flag_pick, ...]

# =============== Allocation and MAE computation =============== #

# N_days
if year%4 == 0:
    N_days = 366
else:
    N_days = 365
    
# other param
N_fcst = 54;
N_grids = BCH_obs.shape[-1]

EN = 75
EN_range = np.arange(5, 76, 1, dtype=int)
N_en = len(EN_range)

# ---------- grided info ---------- #
with h5py.File(save_dir+'NA_SL_info.hdf', 'r') as h5io:
    W_SL = h5io['W_SL'][bc_inds[0]:bc_inds[1], bc_inds[2]:bc_inds[3]][indx, indy]

# ---------- allocations ---------- #
MAE = np.empty((N_days, N_fcst, N_grids, N_en))
SPREAD = np.empty((N_days, N_fcst, N_grids, N_en))
CRPS = np.empty((N_days, N_fcst, N_grids, N_en))

print("Computing CRPS ...")

for lead in range(N_fcst):
    print("lead = {}".format(lead))
    
    with h5py.File(REFCST_dir + "{}_{}_lead{}.hdf".format(perfix_raw, year, lead), 'r') as h5io:
        RAW = h5io[key_raw][:, :EN, ...][..., indx, indy]
            
    with h5py.File(REFCST_dir + "{}_{}_lead{}.hdf".format(perfix_smooth, year, lead), 'r') as h5io:
        SMOOTH = h5io[key_smooth][:, :EN, ...][..., indx, indy]
    
    AnEn = W_SL*RAW + (1-W_SL)*SMOOTH
    
    for i, en in enumerate(EN_range):
    
        crps, mae, _ = metrics.CRPS_1d_nan(BCH_obs[:, lead, :], AnEn[:, :en, ...])
        MAE[:, lead, :, i] = mae
        CRPS[:, lead, :, i] = crps

tuple_save = (MAE, CRPS,)
label_save = ['MAE', 'CRPS',]
du.save_hdf5(tuple_save, label_save, save_dir, '{}_CRPS_vs_EN_{}.hdf'.format(perfix_smooth, year))


