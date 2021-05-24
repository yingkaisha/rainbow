'''
Minimum Divergence Schaake Shuffle (MDSS).

Input: dressed AnEn members.
Output: shuffled AnEn members.
'''

import sys
import time
import argparse

import h5py
import numpy as np
import numba as nb

# # custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/utils/')

from namelist import *
import data_utils as du
import analog_utils as ana
# import model_utils as mu
# import train_utils as tu


@nb.njit()
def schaake_shuffle(fcst, traj):
    '''
    The Schaake shuffle of ensemble `fcst` members
    based on given `traj`ectories.
    '''
    num_traj, N_lead, N_grids = traj.shape
    
    output = np.empty((N_lead, num_traj, N_grids))
    
    for l in range(N_lead):
        for n in range(N_grids):
            
            temp_traj = traj[:, l, n]
            temp_fcst = fcst[l, :, n]
            
            reverse_b_func = np.searchsorted(np.sort(temp_traj), temp_traj)
            
            output[l, :, n] = np.sort(temp_fcst)[reverse_b_func]
    return output

@nb.njit()
def search_nearby_days(day, window=30, leap_year=False):
    '''
    Assuming "day" starts at zero
    Assuming 2*window < 365 
    '''
    if leap_year:
        N_days = 366
        ind_date = np.zeros((N_days,))
    else:
        N_days = 365
        ind_date = np.zeros((N_days,))
        
    ind_right = day+window+1
    ind_left = day-window
    
    if ind_left >= 0 and ind_right <= N_days:
        ind_date[day-window:day+window+1] = True
    elif ind_left < 0:
        ind_date[0:day+window+1] = True
        ind_date[day-window:] = True
    else:
        ind_diff = day+window+1-N_days
        ind_date[day-window:] = True
        ind_date[:ind_diff] = True
    return ind_date

@nb.njit(fastmath=True)
def CDF_estimate(X):
    q_bins = np.array([0.25, 0.5, 0.7, 0.9, 0.95,])  
    _, N_fcst, N_grids = X.shape
    CDF = np.empty((5, N_fcst, N_grids))
    
    for lead in range(N_fcst):
        for n in range(N_grids):
            CDF[:, lead, n] = np.quantile(X[:, lead, n], q_bins)
    return CDF

@nb.njit(fastmath=True)
def total_divergence(CDF1, CDF2):
    _, N_fcst, N_grids = CDF1.shape
    TD = 0
    for lead in range(N_fcst):
        for n in range(N_grids):
            TD += np.sum(np.abs(CDF1[:, lead, n] - CDF2[:, lead, n]))
    return TD

def MD_opt(n_day, fcst_raw, era5_H, N, K, factor=15):
    # initial total divergence
    CDF_fcst = CDF_estimate(np.transpose(fcst_raw[n_day, ...], (1, 0, 2)))
    CDF_era = CDF_estimate(era5_H)
    record = total_divergence(CDF_era, CDF_fcst)

    # stepwise shrinking
    ind_pick = np.arange(N, dtype=np.int_) # all available indices
    
    flag_clean = np.ones((N,), dtype=np.bool_) # output flag
    flag_trial = np.copy(flag_clean)
    
    N_0 = N
    
    flag_single = False
    black_list = []

    while N_0 > K:
        # all available candidate
        N_0 = np.sum(flag_clean)
        ind_candidate = ind_pick[flag_clean]
        #print(N_0)
        
        # randomly selecting candidtes
        size_ = int((N_0-K)/(factor))
        
        if size_ > 5:
            ind_ = np.random.choice(range(N_0), size_, replace=False)
            for i in ind_:
                flag_trial[ind_candidate[i]] = False
        else:
            flag_single = True
            
            j = np.random.choice(ind_candidate, size=1)
            while j in black_list:
                j = np.random.choice(ind_candidate, size=1)
                
            flag_trial[j] = False
            
        era5_sub = era5_H[flag_trial, ...]
        CDF_era = CDF_estimate(era5_sub)
        record_temp = total_divergence(CDF_era, CDF_fcst)

        # stepwise discard
        if record_temp < record:
            #print('hit: {}'.format(record_temp))
            record = record_temp
            if flag_single:
                flag_clean[j] = False
            else:
                flag_clean = np.copy(flag_trial)

        else:
            #print('miss: {}'.format(record_temp))
            if flag_single:
                flag_trial[j] = True
                black_list.append(j)
            else:
                flag_trial = np.copy(flag_clean)
                
    return flag_clean

# =============== MAIN =============== #

# ---------- Parsers ---------- #
parser = argparse.ArgumentParser()
parser.add_argument('out', help='out')
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

type_ind = int(args['out'])
year_fcst = int(args['year'])

if type_ind == 0:
    prefix_raw = 'BASE_final'
    key_raw = 'AnEn'
    
elif type_ind == 1:
    prefix_raw = 'SL_final'
    key_raw = 'AnEn'

elif type_ind == 2:
    prefix_raw = 'BASE_CNN'
    key_raw = 'cnn_pred'

elif type_ind == 3:
    prefix_raw = 'SL_CNN'
    key_raw = 'cnn_pred'

year_analog = np.arange(2000, 2015)

# ---------- Parameters ---------- #
# number of AnEn members: 75
K = 75
EN = 75

# number of lead times: 54 (the end of day-6)
N_fcst = 54

if year_fcst % 4 == 0:
    N_days = 366
else:
    N_days = 365

window_size = 30
W = 2*window_size+1
N = int(len(year_analog)*W)
    
# ---------- Grided info ---------- #
with h5py.File(save_dir+'BC_domain_info.hdf', 'r') as h5io:
    land_mask = h5io['land_mask_base'][...]
    land_mask_bc = h5io['land_mask_bc'][...]
N_grids = np.sum(~land_mask_bc)

# new forecast
print('Loading forecast')
fcst_raw = np.empty((N_days, N_fcst, EN, N_grids))

for lead in range(N_fcst):
    
    with h5py.File(REFCST_dir + "{}_{}_lead{}.hdf".format(prefix_raw, year_fcst, lead), 'r') as h5io:
        RAW = h5io[key_raw][:, :EN, ...]
        
    RAW = RAW[..., ~land_mask_bc]
    fcst_raw[:, lead, :, :] = RAW

fcst_raw[fcst_raw<0] = 0

# reanalysis
print('Loading ERA5')
ERA5 = ()
for year in year_analog: 
    with h5py.File(ERA_dir+'ERA5_GEFS-fcst_{}.hdf'.format(year), 'r') as h5io:
        era_ = h5io['era_fcst'][..., :N_fcst, bc_inds[0]:bc_inds[1], bc_inds[2]:bc_inds[3]]
    ERA5 += (era_[..., ~land_mask_bc],)
    
era5_H = np.empty((N, N_fcst, N_grids))
MDSS = np.empty((365, N_fcst, EN, N_grids))

# ---------- Schaake shuffle ---------- #
print('MDSS starts ...')
start_time = time.time()

for n_day in range(365):
    print("day: {}".format(n_day))
    for i, year in enumerate(year_analog):
        if year % 4 == 0:
            flag_pick = search_nearby_days(n_day, window=window_size, leap_year=True)
        else:
            flag_pick = search_nearby_days(n_day, window=window_size, leap_year=False)

        era5_H[i*W:(i+1)*W, ...] = ERA5[i][flag_pick==1, ...]

    flag_clean = MD_opt(n_day, fcst_raw, era5_H, N, K, factor=10)
    
    era5_traj = era5_H[flag_clean, ...]
    MDSS[n_day, ...] = schaake_shuffle(fcst_raw[n_day, ...], era5_traj)

    print("... Completed. Time = {} sec ".format((time.time() - start_time)))

for l in range(N_fcst):
    tuple_save = (MDSS[:, l, :, :],)
    label_save = [key_raw,]
    du.save_hdf5(tuple_save, label_save, REFCST_dir, '{}_MDSS_{}_lead{}.hdf'.format(prefix_raw, year_fcst, l))
    