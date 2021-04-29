import sys
import time
import os.path
from glob import glob
from datetime import datetime, timedelta

# data tools
import h5py
import numpy as np
import numba as nb

# custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/')

import analog_utils as ana
import data_utils as du
from namelist import * 

lead0 = 0 
lead1 = 54

@nb.njit(fastmath=True)
def CDF_gen(cnn_pred, flag_pick):
    
    N_grids = cnn_pred.shape[-1]
    q_bins = np.array([ 1.0e-04, 5.0e-04, 1.0e-03, 5.0e-03, 1.0e-02, 2.0e-02, 3.0e-02,
                        4.0e-02, 5.0e-02, 6.0e-02, 7.0e-02, 8.0e-02, 9.0e-02, 1.0e-01,
                        1.1e-01, 1.2e-01, 1.3e-01, 1.4e-01, 1.5e-01, 1.6e-01, 1.7e-01,
                        1.8e-01, 1.9e-01, 2.0e-01, 2.1e-01, 2.2e-01, 2.3e-01, 2.4e-01,
                        2.5e-01, 2.6e-01, 2.7e-01, 2.8e-01, 2.9e-01, 3.0e-01, 3.1e-01,
                        3.2e-01, 3.3e-01, 3.4e-01, 3.5e-01, 3.6e-01, 3.7e-01, 3.8e-01,
                        3.9e-01, 4.0e-01, 4.1e-01, 4.2e-01, 4.3e-01, 4.4e-01, 4.5e-01,
                        4.6e-01, 4.7e-01, 4.8e-01, 4.9e-01, 5.0e-01, 5.1e-01, 5.2e-01,
                        5.3e-01, 5.4e-01, 5.5e-01, 5.6e-01, 5.7e-01, 5.8e-01, 5.9e-01,
                        6.0e-01, 6.1e-01, 6.2e-01, 6.3e-01, 6.4e-01, 6.5e-01, 6.6e-01,
                        6.7e-01, 6.8e-01, 6.9e-01, 7.0e-01, 7.1e-01, 7.2e-01, 7.3e-01,
                        7.4e-01, 7.5e-01, 7.6e-01, 7.7e-01, 7.8e-01, 7.9e-01, 8.0e-01,
                        8.1e-01, 8.2e-01, 8.3e-01, 8.4e-01, 8.5e-01, 8.6e-01, 8.7e-01,
                        8.8e-01, 8.9e-01, 9.0e-01, 9.1e-01, 9.2e-01, 9.3e-01, 9.4e-01,
                        9.5e-01, 9.6e-01, 9.7e-01, 9.8e-01, 9.9e-01, 9.95e-01])
    
    CDF = np.empty((12, 105, N_grids))
    flag_pick = flag_pick > 0
    
    for mon in range(12):
        flag_pick_temp = flag_pick[mon, :]
        cnn_sub = cnn_pred[flag_pick_temp, ...]
        
        for n in range(N_grids):
            temp = cnn_sub[..., n].ravel()
            CDF[mon, :-1, n] = np.quantile(temp, q_bins)
            CDF[mon, -1, n] = np.max(temp)
    return CDF

# land mask
with h5py.File(save_dir+'BC_domain_info.hdf', 'r') as h5io:
    land_mask_bc = h5io['land_mask_bc'][...]

q_bins = np.array([ 1.0e-04, 5.0e-04, 1.0e-03, 5.0e-03, 1.0e-02, 2.0e-02, 3.0e-02,
                    4.0e-02, 5.0e-02, 6.0e-02, 7.0e-02, 8.0e-02, 9.0e-02, 1.0e-01,
                    1.1e-01, 1.2e-01, 1.3e-01, 1.4e-01, 1.5e-01, 1.6e-01, 1.7e-01,
                    1.8e-01, 1.9e-01, 2.0e-01, 2.1e-01, 2.2e-01, 2.3e-01, 2.4e-01,
                    2.5e-01, 2.6e-01, 2.7e-01, 2.8e-01, 2.9e-01, 3.0e-01, 3.1e-01,
                    3.2e-01, 3.3e-01, 3.4e-01, 3.5e-01, 3.6e-01, 3.7e-01, 3.8e-01,
                    3.9e-01, 4.0e-01, 4.1e-01, 4.2e-01, 4.3e-01, 4.4e-01, 4.5e-01,
                    4.6e-01, 4.7e-01, 4.8e-01, 4.9e-01, 5.0e-01, 5.1e-01, 5.2e-01,
                    5.3e-01, 5.4e-01, 5.5e-01, 5.6e-01, 5.7e-01, 5.8e-01, 5.9e-01,
                    6.0e-01, 6.1e-01, 6.2e-01, 6.3e-01, 6.4e-01, 6.5e-01, 6.6e-01,
                    6.7e-01, 6.8e-01, 6.9e-01, 7.0e-01, 7.1e-01, 7.2e-01, 7.3e-01,
                    7.4e-01, 7.5e-01, 7.6e-01, 7.7e-01, 7.8e-01, 7.9e-01, 8.0e-01,
                    8.1e-01, 8.2e-01, 8.3e-01, 8.4e-01, 8.5e-01, 8.6e-01, 8.7e-01,
                    8.8e-01, 8.9e-01, 9.0e-01, 9.1e-01, 9.2e-01, 9.3e-01, 9.4e-01,
                    9.5e-01, 9.6e-01, 9.7e-01, 9.8e-01, 9.9e-01, 9.95e-01])

# =================================================== #
freq = 3.0
FCSTs = np.arange(9, 240+freq, freq) # fcst lead as hour

year_hist = np.arange(2000, 2015)

date_start = datetime(2000, 1, 1)
date_end = datetime(2014, 12, 31)

N_days_15y = (date_end-date_start).days
date_list = [date_start + timedelta(days=x) for x in range(N_days_15y+1)]

L = len(date_list)
flag_pick = np.zeros((12, lead1-lead0, L,))

for mon in range(12):
    # ----- #
    if mon == 0:
        month_around = [11, 0, 1] # handle Jan
    elif mon == 11:
        month_around = [10, 11, 0] # handle Dec
    else:
        month_around = [mon-1, mon, mon+1]
    
    for lead in range(lead0, lead1):
        for d, date in enumerate(date_list):
            # ini date + lead time
            date_true = date + timedelta(hours=FCSTs[lead])

            if date_true.month-1 in month_around: 
                flag_pick[mon, lead, d] = 1.0
            else:
                flag_pick[mon, lead, d] = 0.0
            
# =================================================== #
for lead in range(lead0, lead1):
    
    print("lead = {}".format(lead))
    
    gefs = ()
    
    for y in year_hist:
        with h5py.File(REFCST_dir+'En_members_APCP_{}.hdf'.format(y), 'r') as h5io:
            gefs_ = h5io['bc_mean'][:, lead, 0, ...][..., ~land_mask_bc]
        gefs += (gefs_,) # could also use ensemble mean
        
    GEFS = np.concatenate(gefs, axis=0)
    
    start_time = time.time()
    CDF = CDF_gen(GEFS, flag_pick[:, lead, :])
    print("Time = {} sec ".format((time.time() - start_time)))

    tuple_save = (CDF,)
    label_save = ['CDF']
    du.save_hdf5(tuple_save, label_save, save_dir, 'GEFS_lead{}_quantile.hdf'.format(lead))


