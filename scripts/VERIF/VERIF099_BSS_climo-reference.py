
import sys
import argparse
from glob import glob
from datetime import datetime, timedelta

# data tools
import h5py
import pandas as pd
import numba as nb
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/')
sys.path.insert(0, '/glade/u/home/ksha/PUBLISH/fcstpp/')

from fcstpp import metrics, utils
import data_utils as du
from namelist import * 

# Defining params
parser = argparse.ArgumentParser()
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

year = int(args['year'])

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
with h5py.File(ERA_dir+'PT_3hour_q.hdf', 'r') as h5io:
    CDF_era = h5io['era_3hq_bc'][...]
    q_bins = h5io['q_bins'][...]

CDF_obs = np.empty((12, 107,)+land_mask_bc.shape)
CDF_obs[..., ~land_mask_bc] = CDF_era
CDF_obs = CDF_obs[..., indx, indy]

# station and monthly (contains neighbouring months) wise 90th
BCH_90th = CDF_obs[:, (93+9), :] 

N_stn = BCH_obs.shape[-1]

# =============== datetime processing =============== #

# identifying which forecasted day belongs to which month
# thus, the corresponded monthly climo can be pulled.

base = datetime(year, 1, 1)
date_list = [base + timedelta(hours=x) for x in range(N_days)]

flag_pick = np.zeros((N_days, N_fcst), dtype=int)

for d, date in enumerate(date_list):
    for t, fcst_temp in enumerate(FCSTs):
        date_true = date + timedelta(hours=fcst_temp)
        flag_pick[d, t] = date_true.month-1

# =============== BS calculation =============== #

BS_clim = np.empty((N_days, N_fcst, N_stn))

for lead in range(N_fcst):
    for mon in range(12):   
        flag_ = flag_pick[:, lead] == mon
        # stn obs
        obs_ = BCH_obs[flag_, lead, :]
        # station-wise 90-th thresholds
        thres_ = BCH_90th[mon, :] 
        obs_flag = obs_>=thres_
        obs_flag[np.isnan(obs_)] = np.nan
        # BS calculation
        BS_ = (1.0*obs_flag-0.1)**2
        BS_clim[flag_, lead, :] = BS_

# save (all lead times, per year, climatology reference only)
tuple_save = (BS_clim, BCH_90th)
label_save = ['BS', 'stn_99th']
du.save_hdf5(tuple_save, label_save, save_dir, 'CLIM_99th_BS_BCH_{}.hdf'.format(year))

