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
    perfix_raw = 'BASE_final_SS'
    key_raw = 'AnEn'
    EN = 25
    
elif type_ind == 1:
    perfix_raw = 'SL_final_SS'
    key_raw = 'AnEn'
    EN = 25
    
elif type_ind == 2:
    perfix_raw = 'BASE_CNN'
    key_raw = 'AnEn'
    EN = 75 # 25 members dressed to 75
    
elif type_ind == 3:
    perfix_raw = 'SL_BASE'
    key_raw = 'AnEn'
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

# ========== ERA5 stn climatology preprocessing ========== #

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

# number of stations
N_stn = BCH_obs.shape[-1]

# ========== BS computation ========== #

# identifying which forecasted day belongs to which month
# thus, the corresponded monthly climo can be pulled.

base = datetime(year, 1, 1)
date_list = [base + timedelta(hours=x) for x in range(N_days)]

flag_pick = np.zeros((N_days, N_fcst), dtype=int)

for d, date in enumerate(date_list):
    for t, fcst_temp in enumerate(FCSTs):
        date_true = date + timedelta(hours=fcst_temp)
        flag_pick[d, t] = date_true.month-1

# ---------- allocations ---------- #

# allocation
BS = np.empty((N_days, N_fcst, N_stn))

for lead in range(N_fcst):
    print("computing lead: ".format(lead))
    
    with h5py.File(REFCST_dir + "{}_{}_lead{}.hdf".format(perfix_raw, year, lead), 'r') as h5io:
        AnEn_stn = h5io[key_raw][:, :EN, ...][..., indx, indy]

    # extracting the 90-th threshold for initializaiton time + lead time
    for mon in range(12):
        flag_ = flag_pick[:, lead] == mon
        # stn obs
        obs_ = BCH_obs[flag_, lead, :]
        # fcst
        pred_ = AnEn_stn[flag_, ...]
        # station-wise threshold
        thres_ = BCH_90th[mon, :]
        # Brier Score ( ">=" is applied)
        BS[flag_, lead, :] = metrics.BS_binary_1d_nan((obs_>=thres_), (pred_>=thres_))

# save (all lead times, per year, per experiment)
tuple_save = (BS, BCH_90th)
label_save = ['BS', 'stn_90th']
du.save_hdf5(tuple_save, label_save, save_dir, '{}_BS_BCH_{}.hdf'.format(perfix_smooth, year))
