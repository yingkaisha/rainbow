
# sys tools
import sys
import time
import argparse

# data tools
import h5py
import numpy as np
import numba as nb

# custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/utils/')

import data_utils as du
import analog_utils as ana

from namelist import * 

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
def analog_search(day0, day1, year_analog, fcst_apcp, fcst_pwat, APCP, PWAT, ERA5, IND_bc):
    
    # params that can be adjusted
    EN = 75 # number of AnEn members
    N_grids = fcst_apcp.shape[1] # number of grid points
    window_day = 30 # time window (by days) of the analog search 
    shape_ravel = (2*window_day+1,)
    
    # number of initialization days
    L_fcst_days = day1 - day0
    
    # output
    AnEn = np.empty((L_fcst_days, N_grids, EN))

    # allocate single day, grid, and year
    day_n = np.empty((EN,), np.int_)
    ind_n = np.empty((EN,), np.int_)
    year_n = np.empty((EN,), np.int_)
    record_n = np.ones((EN,))
    
    # datetime related variables
    day_365 = np.arange(365)
    day_366 = np.arange(366)

    # loop over initialization time
    for day_i, day_new in enumerate(range(day0, day1)):
        
        # loop over grid points
        for n in range(N_grids):
            # fcst values
            apcp_new = fcst_apcp[day_new, n]
            pwat_new = fcst_pwat[day_new, n]
            
            # initialize analog selction records
            record_n[:] = 9999
            
            # loop over reforecast
            for year_ind, year_ana in enumerate(year_analog):
                
                # the 91-day window of analog search
                if year_ana%4 == 0:
                    flag_analog_days = search_nearby_days(day_new, window=window_day, leap_year=True)
                    day_base = day_366[flag_analog_days==1]
                    
                else:
                    flag_analog_days = search_nearby_days(day_new, window=window_day, leap_year=False)
                    day_base = day_365[flag_analog_days==1]
                    
                # loop over a single year reforecast
                for d in range(shape_ravel[0]):
                    
                    day_real = int(day_base[d])
                    apcp_old = APCP[year_ind][day_real, n]
                    pwat_old = PWAT[year_ind][day_real, n]

                    # analog criteria
                    record_temp = 0.76*np.abs(apcp_old - apcp_new) + 0.24*np.abs(pwat_old - pwat_new)
                        
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
                            
            # back to the grid point loop
            # assigning ERA5 based on the (multi-year) reforecast search
            for en in range(EN):
                AnEn[day_i, n, en] = ERA5[year_n[en]][day_n[en], n]
    
    # AnEn.shape = (L_fcst_days, N_grids, EN)
    return AnEn

# -------- Function ends -------- #

parser = argparse.ArgumentParser()
parser.add_argument('year_fcst', help='year_fcst')
parser.add_argument('part', help='part')
args = vars(parser.parse_args())

# ---------------------------------- #
# arg1
year_fcst = int(args['year_fcst'])
print("Applying year {} GEFS for testing".format(year_fcst))

day0 = 0 # <----
if year_fcst%4 == 0:
    day1 = 366
    flag_leap_year = True
else:
    day1 = 365
    flag_leap_year = False
    
L_fcst_days = day1 - day0

# ---------------------------------- #
# arg2
part_ = int(args['part'])
print("Part {}".format(part_))

if part_ == 0:
    LEADs = np.arange(0, 27, dtype=np.int)
else:
    LEADs = np.arange(27, 54, dtype=np.int)

# ---------------------------------- #

EN = 75
year_analog = np.arange(2000, 2015)

# importing domain information
with h5py.File(save_dir+'BC_domain_info.hdf', 'r') as h5io:
    land_mask = h5io['land_mask_base'][...]
    land_mask_bc = h5io['land_mask_bc'][...]
bc_in_base = np.ones(land_mask.shape).astype(bool)
bc_in_base[bc_inds[0]:bc_inds[1], bc_inds[2]:bc_inds[3]] = land_mask_bc

# subsetting by land mask
bc_shape = land_mask_bc.shape
grid_shape = land_mask.shape
IND_bc = []
for i in range(grid_shape[0]):
    for j in range(grid_shape[1]):
        if ~bc_in_base[i, j]:
            IND_bc.append([i, j])
IND_bc = np.array(IND_bc, dtype=np.int)
N_grids = len(IND_bc)

for lead in LEADs:
    print("Processing lead time = {}".format(lead))
    # ------------------------------------------------- #
    # Import reforecast
    APCP = ()
    PWAT = ()
    for year in year_analog:
        with h5py.File(REFCST_dir+'En_mean_APCP_{}.hdf'.format(year), 'r') as h5io:
            apcp_temp = h5io['base_mean'][:, lead, ...][..., IND_bc[:, 0], IND_bc[:, 1]]
        with h5py.File(REFCST_dir+'En_mean_PWAT_{}.hdf'.format(year), 'r') as h5io:
            pwat_temp = h5io['base_mean'][:, lead, ...][..., IND_bc[:, 0], IND_bc[:, 1]]
        APCP += (apcp_temp,)
        PWAT += (pwat_temp,)
    
    # ------------------------------------------------- #
    # Import reanalysis
    ERA5 = ()
    for year in year_analog:
        with h5py.File(ERA_dir+'ERA5_GEFS-fcst_{}.hdf'.format(year), 'r') as h5io:
            era_temp = h5io['era_fcst'][:, lead, ...][..., IND_bc[:, 0], IND_bc[:, 1]]
        ERA5 += (era_temp,)
        
    # ------------------------------------------------- #
    # importing new fcst
    with h5py.File(REFCST_dir+'En_mean_APCP_{}.hdf'.format(year_fcst), 'r') as h5io:
        fcst_apcp = h5io['base_mean'][:, lead, ...][..., IND_bc[:, 0], IND_bc[:, 1]]
    with h5py.File(REFCST_dir+'En_mean_PWAT_{}.hdf'.format(year_fcst), 'r') as h5io:
        fcst_pwat = h5io['base_mean'][:, lead, ...][..., IND_bc[:, 0], IND_bc[:, 1]]

    # ------------------------------------------------- #
    print("AnEn search starts ...")
    start_time = time.time()
    AnEn = analog_search(day0, day1, year_analog, fcst_apcp, fcst_pwat, APCP, PWAT, ERA5, IND_bc)
    print("... Completed. Time = {} sec ".format((time.time() - start_time)))
    
#     # ------------------------------------------------- #
#     print("SG filter starts ...")
#     start_time2 = time.time()
    AnEn_grid = np.empty((L_fcst_days, EN)+bc_shape)
#     AnEn_grid[...] = 0.0

#     AnEn_SG = np.empty((L_fcst_days, EN)+bc_shape)
#     AnEn_SG[...] = np.nan

    for i in range(L_fcst_days):
        for j in range(EN):
            AnEn_grid[i, j, ~land_mask_bc] = AnEn[i, ..., j]
#             # smoothings
#             temp_ = AnEn_grid[i, j, ...]
#             temp_barnes = ana.sg2d(temp_, window_size=9, order=3, derivative=None) # <-- copied
#             temp_barnes[~land_mask_bc] = temp_[~land_mask_bc]
#             temp_barnes = ana.sg2d(temp_barnes, window_size=9, order=3, derivative=None) # <-- copied
#             temp_barnes[land_mask_bc] = np.nan
#             AnEn_SG[i, j, ...] = temp_barnes

    AnEn_grid[..., land_mask_bc] = np.nan
#     print("... Completed. Time = {} sec ".format((time.time() - start_time2)))  
#     tuple_save = (AnEn_grid, AnEn_SG)
#     label_save = ['AnEn', 'AnEn_SG']
    
    tuple_save = (AnEn_grid,)
    label_save = ['AnEn',]
    du.save_hdf5(tuple_save, label_save, REFCST_dir, 'BASE_final_{}_lead{}.hdf'.format(year_fcst, lead))
    
