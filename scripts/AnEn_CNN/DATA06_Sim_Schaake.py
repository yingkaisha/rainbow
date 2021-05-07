'''
Similarity-based implementation of Schaake Shuffle.

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
# sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/utils/')

from namelist import *
import data_utils as du
# import model_utils as mu
# import train_utils as tu

@nb.njit()
def shift_one(arr):
    arr[1:] = arr[:-1]
    return arr

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
def sim_schaake(year_analog, fcst_en, fcst_raw, APCP, ERA5, weights):
    '''
    Similarity-based Schaake Shuffle.
    
    Input
    ----------
    year_analog: the correspondded initialization year of AnEn members.
    fcst_en: AnEn members used for similarity search (can be the same as `fcst_raw`).
    fcst_raw: AnEn members to be shuffled.
    APCP: the GEFS reforecast.
    ERA5: the shuffling targets (trajectories).
    weights: forecast lead time dependent weights 
             (similarities in short lead times are more important).
    
    Output
    ----------
    AnEn: shuffled AnEn members.
    '''
    
    # begin from the first day of `year_analog`
    day0 = 0 

    # initialization time, lead time, number of members, number of grids
    day1, N_leads, EN, N_grids = fcst_en.shape
    
    # number of initialization days
    L_fcst_days = day1 - day0
    
    # time window (by days) of the similarity search 
    window_day = 30 
    shape_ravel = (2*window_day+1,)
    
    # allocations
    fcst_apcp = np.empty((L_fcst_days, N_leads, N_grids))
    ERA5_traj = np.empty((EN, N_leads, N_grids))
    AnEn = np.empty((L_fcst_days, N_leads, EN, N_grids))

    # allocate single day, grid, and year
    day_n = np.empty((EN,), np.int_)
    ind_n = np.empty((EN,), np.int_)
    year_n = np.empty((EN,), np.int_)
    record_n = np.ones((EN,))
    
    # days to indices
    day_365 = np.arange(365)
    day_366 = np.arange(366)
    
    # compute ensemble mean for the similarity search
    # i.e., matching the AnEn-mean and reforecast-ensemble-mean to identify trajectories
    for day_new in range(L_fcst_days):
        for l in range(N_leads):
            for n in range(N_grids):
                fcst_apcp[day_new, l, n] = np.mean(fcst_en[day_new, l, :, n])
    
    # loop over initialization time
    for day_i, day_new in enumerate(range(day0, day1)):
        
        # fcst values shape=(N_leads, N_grids)
        apcp_new = fcst_apcp[day_new, :, :]
        
        # initialize similarity selction records
        record_n[:] = 9999
            
        # loop over all possible reforecast years.
        for year_ind, year_ana in enumerate(year_analog):
            
            # the `2*window+1` days of the similarity search
            if year_ana%4 == 0:
                flag_analog_days = search_nearby_days(day_new, window=window_day, leap_year=True)
                day_base = day_366[flag_analog_days==1]

            else:
                flag_analog_days = search_nearby_days(day_new, window=window_day, leap_year=False)
                day_base = day_365[flag_analog_days==1]
                    
            # loop over a single year reforecast
            for d in range(shape_ravel[0]):

                day_real = int(day_base[d])
                
                # ** similarity measures ** #
                record_temp = 0
                for l in range(N_leads):
                    for n in range(N_grids):
                        record_temp += weights[l]*np.abs(APCP[year_ind][day_real, l, n] - apcp_new[l, n])
                record_temp = record_temp/N_grids/N_leads

                # if hit the new record
                if record_temp < record_n[-1]:

                    # searchosrt positions
                    ind_analog = np.searchsorted(record_n, record_temp)

                    # shift one from the position to free space
                    day_n[ind_analog:] = shift_one(day_n[ind_analog:])
                    year_n[ind_analog:] = shift_one(year_n[ind_analog:])
                    record_n[ind_analog:] = shift_one(record_n[ind_analog:])

                    # insert
                    day_n[ind_analog] = day_real
                    year_n[ind_analog] = year_ind
                    record_n[ind_analog] = record_temp

        
        for en in range(EN):
            ERA5_traj[en, ...] = ERA5[year_n[en]][day_n[en], :, :]
            
        #AnEn[day_i, ...] = schaake_shuffle(fcst_en[day_new, ...], ERA5_traj)
        AnEn[day_i, ...] = schaake_shuffle(fcst_raw[day_new, ...], ERA5_traj)
        
    return AnEn

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

# similarity search period: 2000-2014 (15 years)
year_analog = np.arange(2000, 2015)

# ---------- Parameters ---------- #
# number of AnEn members: 75
EN = 75
# number of lead times: 54 (the end of day-6)
N_fcst = 54

if year_fcst % 4 == 0:
    N_days = 366
else:
    N_days = 365

# weights (higher for shorter lead times)
N_leads = N_fcst
# lead time weights
x = np.linspace(0, 0.05*N_leads, N_leads)
y = np.exp(x); y_max = np.max(y)
y = y/y_max; weights = (1-y)
    
# ---------- Grided info ---------- #
with h5py.File(save_dir+'BC_domain_info.hdf', 'r') as h5io:
    land_mask = h5io['land_mask_base'][...]
    land_mask_bc = h5io['land_mask_bc'][...]
N_grids = np.sum(~land_mask_bc)

# reforecast
APCP = ()
for year in year_analog:
    with h5py.File(REFCST_dir+'En_mean_APCP_{}.hdf'.format(year), 'r') as h5io:
        apcp_temp = h5io['bc_mean'][:, :N_fcst, ...]
    APCP += (apcp_temp[..., ~land_mask_bc],)
    
# reanalysis
ERA5 = ()
for year in year_analog: 
    with h5py.File(ERA_dir+'ERA5_GEFS-fcst_{}.hdf'.format(year), 'r') as h5io:
        era_ = h5io['era_fcst'][..., :N_fcst, bc_inds[0]:bc_inds[1], bc_inds[2]:bc_inds[3]]
    ERA5 += (era_[..., ~land_mask_bc],)

# new forecast
fcst_ref = np.empty((N_days, N_fcst, EN, N_grids))
fcst_raw = np.empty((N_days, N_fcst, EN, N_grids))

for lead in range(N_fcst):
    
    with h5py.File(REFCST_dir + "{}_{}_lead{}.hdf".format(prefix_raw, year_fcst, lead), 'r') as h5io:
        RAW = h5io[key_raw][:, :EN, ...]
    RAW = RAW[..., ~land_mask_bc]

    fcst_ref[:, lead, :, :] = RAW
    fcst_raw[:, lead, :, :] = RAW

fcst_ref[fcst_ref<0] = 0
fcst_raw[fcst_raw<0] = 0

# ---------- Schaake shuffle ---------- #
print('SimSchaake starts ...')
start_time = time.time()
fcst_ss = sim_schaake(year_analog, fcst_ref, fcst_raw, APCP, ERA5, weights)
print("... Completed. Time = {} sec ".format((time.time() - start_time)))

for l in range(N_fcst):
    tuple_save = (fcst_ss[:, l, :, :],)
    label_save = [key_raw,]
    du.save_hdf5(tuple_save, label_save, REFCST_dir, '{}_SS_{}_lead{}.hdf'.format(prefix_raw, year_fcst, l))
    
