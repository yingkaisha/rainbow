
import sys
import argparse
from glob import glob
from datetime import datetime, timedelta

# data tools
import h5py
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

# Importing Geo-info
with h5py.File(save_dir+'BC_domain_info.hdf', 'r') as h5io:
    land_mask_bc = h5io['land_mask_bc'][...]

# Defining params
parser = argparse.ArgumentParser()
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

year_target = int(args['year'])
year_clim = np.arange(2000, 2015)

if year_target%4 == 0:
    N_days = 366
else:
    N_days = 365

N_fcst = 54

period = 3
feb_29 = int((31+28)*24/period)
delta_day = int(24/period)

FCSTs = np.arange(9.0, 24*9+period, period)
N_grids = np.sum(~land_mask_bc)

# Importing ERA5 obs
with h5py.File(ERA_dir+'PT_3hr_{}.hdf'.format(year_target), 'r') as h5io:
    era_obs1 = h5io['era_025'][...][..., bc_inds[0]:bc_inds[1], bc_inds[2]:bc_inds[3]]
with h5py.File(ERA_dir+'PT_3hr_{}.hdf'.format(year_target+1), 'r') as h5io:
    era_obs2 = h5io['era_025'][...][..., bc_inds[0]:bc_inds[1], bc_inds[2]:bc_inds[3]]
era_obs = np.concatenate((era_obs1[:, ~land_mask_bc], era_obs2[:, ~land_mask_bc]), axis=0)

# Importing ERA5 climatology
ERA5 = {}
for year in year_clim:
    with h5py.File(ERA_dir+'PT_3hr_{}.hdf'.format(year), 'r') as h5io:
        era_temp = h5io['era_025'][...][..., bc_inds[0]:bc_inds[1], bc_inds[2]:bc_inds[3]][:, ~land_mask_bc]
        ERA5['{}'.format(year)] = era_temp
ERA5 = tuple(ERA5.values())

# ERA quantiles
with h5py.File(ERA_dir+'PT_3hour_quantile.hdf', 'r') as h5io:
    CDF_era = h5io['CDF'][...]
    q_bins = h5io['q'][...]

# Brier Score calculations on pre-defined quantiles
for p in np.array([0.9,]):
    
    print('Computing BS on quantile values of {}'.format(p))
    
    quantile_grids = CDF_era[:, :-1, :]
    quantile_grids = quantile_grids[:, q_bins==p, :]

    # converting each historical years into flags
    # these flags will be averaged as historical probabilities
    for y in range(len(year_clim)):

        era_ = quantile_to_flag(ERA5[y], quantile_grids, year=year_clim[y])

        if y == 0:
            ERA5_cate = (era_,)
        else:
            ERA5_cate += (era_,)

    # converting obs into flags      
    era_cate = quantile_to_flag(era_obs, quantile_grids, year=year_target)

    # calculating probabilities from flags
    clim_prob_366, _ = utils.climate_subdaily_prob(ERA5_cate, day_window=30, period=3)

    # match clim and obs
    # ---------- #
    clim_prob_365 =  np.concatenate((clim_prob_366[:feb_29, :], clim_prob_366[feb_29+delta_day:, :]), axis=0)

    if year_target%4 == 0:
        clim_prob = np.concatenate((clim_prob_366, clim_prob_365), axis=0)
    elif (year_target+1)%4 == 0:
        clim_prob = np.concatenate((clim_prob_365, clim_prob_366), axis=0)
    else:
        clim_prob = np.concatenate((clim_prob_365, clim_prob_365), axis=0)

    L = len(era_cate)
    clim_prob = clim_prob[:L, :]
    # ---------- #

    UNC = np.sum((clim_prob-era_cate)**2, axis=1)

    UNC_leads = np.empty((N_days, len(FCSTs),))
    UNC_leads[...] = np.nan

    # Converting datetime to initialization time and forecast leads
    for day in range(N_days):
        for t, fcst_temp in enumerate(FCSTs):
            date_true = datetime(year_target, 1, 1)+timedelta(days=day)+timedelta(hours=fcst_temp)
            year_true = int(date_true.year)
            ind_true = int((date_true-datetime(year_true, 1, 1)).total_seconds()/60/60/3.0)
            UNC_leads[day, t] = UNC[ind_true,]

    UNC_leads = UNC_leads[:, :N_fcst]
    
    tuple_save = (UNC_leads,)
    label_save = ['BS']
    du.save_hdf5(tuple_save, label_save, save_dir, 'ERA_BS-{}th_clim_{}.hdf'.format(int(100*p), year_target))

    
# Brier Score calculations on pre-defined thresholds
for THRES in np.array([5, 30.0]):
    
    thres = THRES/8.0
    
    print("Computing BS on thres of {} mm/day, {} mm/3h".format(THRES, thres))
    
    for y in range(len(year_clim)):
        if y == 0:
            ERA5_cate = ((ERA5[y] > thres),)
        else:
            ERA5_cate += ((ERA5[y] > thres),)
            
    era_cate = era_obs > thres
    
    # calculating probabilities from flags
    clim_prob_366, _ = utils.climate_subdaily_prob(ERA5_cate, day_window=30, period=3)

    # match clim and obs
    # ---------- #
    clim_prob_365 =  np.concatenate((clim_prob_366[:feb_29, :], clim_prob_366[feb_29+delta_day:, :]), axis=0)

    if year_target%4 == 0:
        clim_prob = np.concatenate((clim_prob_366, clim_prob_365), axis=0)
    elif (year_target+1)%4 == 0:
        clim_prob = np.concatenate((clim_prob_365, clim_prob_366), axis=0)
    else:
        clim_prob = np.concatenate((clim_prob_365, clim_prob_365), axis=0)

    L = len(era_cate)
    clim_prob = clim_prob[:L, :]
    # ---------- #
    
    UNC = np.sum((clim_prob-era_cate)**2, axis=1)

    UNC_leads = np.empty((N_days, len(FCSTs),))
    UNC_leads[...] = np.nan

    # Converting datetime to initialization time and forecast leads
    for day in range(N_days):
        for t, fcst_temp in enumerate(FCSTs):
            date_true = datetime(year_target, 1, 1)+timedelta(days=day)+timedelta(hours=fcst_temp)
            year_true = int(date_true.year)
            ind_true = int((date_true-datetime(year_true, 1, 1)).total_seconds()/60/60/3.0)
            UNC_leads[day, t] = UNC[ind_true,]

    UNC_leads = UNC_leads[:, :N_fcst]
    
    tuple_save = (UNC_leads,)
    label_save = ['BS']
    du.save_hdf5(tuple_save, label_save, save_dir, 'ERA_BS-{}mm_clim_{}.hdf'.format(int(THRES), year_target))
    
    