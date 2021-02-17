
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

# Importing Geo-info
with h5py.File(save_dir+'BC_domain_info.hdf', 'r') as h5io:
    land_mask_bc = h5io['land_mask_bc'][...]

# Defining params
parser = argparse.ArgumentParser()
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

year_target = int(args['year'])

if year_target%4 == 0:
    N_days = 366
else:
    N_days = 365

N_fcst = 54

period = 3
delta_day = int(24/period)
FCSTs = np.arange(9.0, 24*9+period, period)
N_grids = np.sum(~land_mask_bc)

# =============== CDF processing =============== #
# ERA quantiles
with h5py.File(ERA_dir+'PT_3hour_quantile.hdf', 'r') as h5io:
    CDF_era = h5io['CDF'][...]

N_mon, _, N_grids = CDF_era.shape

# re-compute quantile bins to {0, 0.01, 0.99, 1.00}. 1.00 is the max
q_bins = np.arange(0, 1.01, 0.01)

CDF_clim = np.empty((N_mon, 101, N_grids))

for i in range(N_mon):
    for j in range(N_grids):
        CDF_clim[i, :, j] = np.concatenate((np.array([CDF_era[0, 0, 0]]), CDF_era[0, 4:-2, 0], np.array([CDF_era[0, -1, 0]])))
        
# better drop the max val
CDF_clim = CDF_clim[:, :-1, :]
q_bins = q_bins[:-1]

# =============== Importing ERA5 obs =============== #

with h5py.File(ERA_dir+'PT_3hr_{}.hdf'.format(year_target), 'r') as h5io:
    era_obs1 = h5io['era_025'][...][..., bc_inds[0]:bc_inds[1], bc_inds[2]:bc_inds[3]]
with h5py.File(ERA_dir+'PT_3hr_{}.hdf'.format(year_target+1), 'r') as h5io:
    era_obs2 = h5io['era_025'][...][..., bc_inds[0]:bc_inds[1], bc_inds[2]:bc_inds[3]]
era_obs = np.concatenate((era_obs1[:, ~land_mask_bc], era_obs2[:, ~land_mask_bc]), axis=0)

# =============== datetime processing =============== #
N_3h = len(era_obs)

base = datetime(year_target, 1, 1)
date_list = [base + timedelta(hours=x) for x in range(0, N_3h*3, 3)]

flag_pick = np.zeros((N_3h,), dtype=int)

for d, date in enumerate(date_list):        
    flag_pick[d] = date.month-1

# =============== CRPS calculation =============== #    
CRPS_clim = np.empty((N_3h, N_grids))

for mon in range(12):
    flag_ = flag_pick == mon
    
    crps_ = metrics.CRPS_1d_from_quantiles(q_bins, CDF_clim[mon, ...], era_obs[flag_, :])
    
    CRPS_clim[flag_, :] = crps_
    
CRPS = np.empty((N_days, len(FCSTs),)+land_mask_bc.shape)
CRPS[...] = np.nan

# Converting datetime to initialization time and forecast leads
for day in range(N_days):
    for t, fcst_temp in enumerate(FCSTs):
        date_true = datetime(year_target, 1, 1) + timedelta(days=day) + timedelta(hours=fcst_temp)
        year_true = int(date_true.year)
        ind_true = int((date_true-datetime(year_true, 1, 1)).total_seconds()/60/60/3.0)
        CRPS[day, t, ~land_mask_bc] = CRPS_clim[ind_true, :]

CRPS = CRPS[:, :N_fcst, :]
    
# Save    
tuple_save = (CRPS,)
label_save = ['CRPS']
du.save_hdf5(tuple_save, label_save, save_dir, 'CLIM_CRPS_{}.hdf'.format(year_target))

    
