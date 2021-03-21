
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

def quantile_to_flag(data, quantile_grids, year, period=3):
    '''
    determine if gridded fcst/obs is larger than 
    the given grid-point-wise quantile values.  
    '''
    
    input_shape = data.shape
    output = np.empty(input_shape)
    output[...] = np.nan
    
    if year%4 == 0:
        N_days = 366
    else:
        N_days = 365
    
    base = datetime(year, 1, 1)
    date_list = [base + timedelta(hours=x) for x in range(0, N_days*24, period)]
    
    if len(input_shape) == 2:    
        for d, date in enumerate(date_list):
            mon_ind = date.month-1
            output[d, :] = data[d, :] > quantile_grids[mon_ind, :]
    else:
        for d, date in enumerate(date_list):
            mon_ind = date.month-1
            for en in range(input_shape[1]):
                output[d, en, :] = data[d, en, :] > quantile_grids[mon_ind, :]
    return output

# Defining params
parser = argparse.ArgumentParser()
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

year = int(args['year'])

N_fcst = 54
period = 3
feb_29 = int((31+28)*24/period)
delta_day = int(24/period)

FCSTs = np.arange(9.0, 24*9+period, period)
FCSTs = FCSTs[:N_fcst]

# ========== BCH obs preprocessing ========== # 

# import station obsevations and grid point indices
with h5py.File(save_dir+'BCH_ERA5_3H_verif.hdf', 'r') as h5io:
    BCH_series = h5io['BCH_obs_series'][...]
    indx = h5io['indx'][...]
    indy = h5io['indy'][...]
    
# subsetting BCH obs into a given year
N_days = 366 + 365*3 +366
date_base = datetime(2016, 1, 1)
date_list = [date_base + timedelta(hours=x) for x in np.arange(0, N_days*24, period, dtype=np.float)]

flag_pick = []
for date in date_list:
    if date.year == year or date.year == year+1:
        flag_pick.append(True)
    else:
        flag_pick.append(False)

flag_pick = np.array(flag_pick)
    
BCH_series = BCH_series[flag_pick, ...]

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

year_clim = np.arange(2000, 2015)

# Importing ERA5 climatology
ERA5 = ()
ERA5_cate = ()

for y in year_clim:
    with h5py.File(ERA_dir+'PT_3hr_{}.hdf'.format(y), 'r') as h5io:
        era_temp = h5io['era_025'][...][..., bc_inds[0]:bc_inds[1], bc_inds[2]:bc_inds[3]][:, indx, indy]
        ERA5 += (era_temp,)
        
for y in range(len(year_clim)):
    era_ = quantile_to_flag(ERA5[y], BCH_90th, year=year_clim[y])
    ERA5_cate += (era_,)

# =============== datetime processing =============== #
print(year)

# N days
if year%4 == 0:
    N_days = 366
else:
    N_days = 365

# # identifying which forecasted day belongs to which month
# # thus, the corresponded monthly climo can be pulled.

# base = datetime(year, 1, 1)
# date_list = [base + timedelta(hours=x) for x in range(N_days)]

# flag_pick = np.zeros((N_days, N_fcst), dtype=int)

# for d, date in enumerate(date_list):
#     for t, fcst_temp in enumerate(FCSTs):
#         date_true = date + timedelta(hours=fcst_temp)
#         flag_pick[d, t] = date_true.month-1

# =============== BS calculation =============== #

# converting obs into flags      
BCH_cate = quantile_to_flag(BCH_series, BCH_90th, year=year)

# converting 2000-2014 ERA5 into flags   
# calculating probabilities from flags
clim_prob_366, _ = utils.climate_subdaily_prob(ERA5_cate, day_window=30, period=3)

# match clim and obs
# ---------- #
clim_prob_365 =  np.concatenate((clim_prob_366[:feb_29, :], clim_prob_366[feb_29+delta_day:, :]), axis=0)

if year%4 == 0:
    clim_prob = np.concatenate((clim_prob_366, clim_prob_365), axis=0)
elif (year+1)%4 == 0:
    clim_prob = np.concatenate((clim_prob_365, clim_prob_366), axis=0)
else:
    clim_prob = np.concatenate((clim_prob_365, clim_prob_365), axis=0)

# ---------- #
UNC = np.sum((clim_prob-BCH_cate)**2, axis=1)

UNC_leads = np.empty((N_days, len(FCSTs),))
UNC_leads[...] = np.nan

# Converting datetime to initialization time and forecast leads
for day in range(N_days):
    for t, fcst_temp in enumerate(FCSTs):
        date_true = datetime(year, 1, 1)+timedelta(days=day)+timedelta(hours=fcst_temp)
        year_true = int(date_true.year)
        ind_true = int((date_true-datetime(year_true, 1, 1)).total_seconds()/60/60/3.0)
        UNC_leads[day, t] = UNC[ind_true,]

UNC_leads = UNC_leads[:, :N_fcst]
    
tuple_save = (UNC_leads, BCH_90th)
label_save = ['BS', 'stn_90th']
du.save_hdf5(tuple_save, label_save, save_dir, 'CLIM_BS_BCH_{}.hdf'.format(year))

