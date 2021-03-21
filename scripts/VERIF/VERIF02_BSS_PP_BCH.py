import sys
import argparse
from glob import glob
from datetime import datetime, timedelta

# data tools
import h5py
import pandas as pd
import numba as nb
import numpy as np

# custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/')
sys.path.insert(0, '/glade/u/home/ksha/PUBLISH/fcstpp/')

from fcstpp import metrics, utils
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
    perfix_smooth = 'BASE_CNN_QM'
    perfix_raw = 'BASE_final'
    key_smooth = 'cnn_pred'
    key_raw = 'AnEn'
    
elif type_ind == 3:
    perfix_smooth = 'SL_CNN_QM'
    perfix_raw = 'SL_final'
    key_smooth = 'cnn_pred'
    key_raw = 'AnEn'

print("perfix_smooth = {}; perfix_raw = {}".format(perfix_smooth, perfix_raw))

N_fcst = 54
period = 3

FCSTs = np.arange(9.0, 24*9+period, period)
FCSTs = FCSTs[:N_fcst]

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

# # ========== ERA5 stn climatology preprocessing ========== #

# importing domain info
with h5py.File(save_dir+'BC_domain_info.hdf', 'r') as h5io:
    land_mask_bc = h5io['land_mask_bc'][...]

# importing girdded ERA5 quantiles
with h5py.File(ERA_dir+'PT_3hour_quantile.hdf', 'r') as h5io:
    CDF_era = h5io['CDF'][...]
    q_bins = h5io['q'][...]

CDF_obs = np.empty((12, 105,)+land_mask_bc.shape)
CDF_obs[..., ~land_mask_bc] = CDF_era
CDF_obs = CDF_obs[..., indx, indy]

# station and monthly (contains neighbouring months) wise 90th
BCH_90th = CDF_obs[:, 93, :] 

# ========== BS computation ========== #

# N_days
if year%4 == 0:
    N_days = 366
else:
    N_days = 365

# other param
EN = 75
N_stn = BCH_obs.shape[-1]
    
# identifying which forecasted day belongs to which month
# thus, the corresponded monthly climo can be pulled.

base = datetime(year, 1, 1)
date_list = [base + timedelta(hours=x) for x in range(N_days)]

flag_pick = np.zeros((N_days, N_fcst), dtype=int)

for d, date in enumerate(date_list):
    for t, fcst_temp in enumerate(FCSTs):
        date_true = date + timedelta(hours=fcst_temp)
        flag_pick[d, t] = date_true.month-1

# ---------- grided info ---------- #
with h5py.File(save_dir+'NA_SL_info.hdf', 'r') as h5io:
    W_SL = h5io['W_SL'][bc_inds[0]:bc_inds[1], bc_inds[2]:bc_inds[3]][indx, indy]

# ---------- allocations ---------- #
BS = np.empty((N_days, N_fcst, N_stn))

print("Computing BS ...")

for lead in range(N_fcst):
    print("lead = {}".format(lead))
    
    with h5py.File(REFCST_dir + "{}_{}_lead{}.hdf".format(perfix_raw, year, lead), 'r') as h5io:
        RAW = h5io[key_raw][:, :EN, ...][..., indx, indy]
    
    with h5py.File(REFCST_dir + "{}_{}_lead{}.hdf".format(perfix_smooth, year, lead), 'r') as h5io:
        SMOOTH = h5io[key_smooth][:, :EN, ...][..., indx, indy]
    
    AnEn = W_SL*RAW + (1-W_SL)*SMOOTH

    for mon in range(12):
        
        flag_ = flag_pick[:, lead] == mon
        
        obs_ = BCH_obs[flag_, lead, :]
        pred_ = AnEn[flag_, ...]
        
        thres_ = BCH_90th[mon, :] # station-wise 90-th vals
        
        BS[flag_, lead, :] = metrics.BS_binary_1d_nan((obs_>thres_), (pred_>thres_))

tuple_save = (BS, BCH_90th)
label_save = ['BS', 'stn_90th']
du.save_hdf5(tuple_save, label_save, save_dir, '{}_BS_BCH_{}.hdf'.format(perfix_smooth, year))
