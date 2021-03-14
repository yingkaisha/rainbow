'''
Computing the CRPS and MAE of BCH obs and raw AnEn outputs. 
'''

import sys
import argparse
from glob import glob
from datetime import datetime, timedelta

# data tools
import h5py
import time
import numba as nb
import numpy as np

# custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/')
sys.path.insert(0, '/glade/u/home/ksha/PUBLISH/fcstpp/')

from fcstpp import metrics
import data_utils as du
from namelist import * 

parser = argparse.ArgumentParser()
parser.add_argument('out', help='out')
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

type_ind = int(args['out'])
year = int(args['year'])

if type_ind == 0:
    perfix_raw = 'BASE_final'
    key_raw = 'AnEn'
    
elif type_ind == 1:
    perfix_raw = 'SL_final'
    key_raw = 'AnEn'

elif type_ind == 2:
    perfix_raw = 'BASE_APCP'
    key_raw = 'AnEn'
    
elif type_ind == 3:
    perfix_raw = 'BASE_PWAT'
    key_raw = 'AnEn'
    
print("perfix_raw = {}".format(perfix_raw))

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

# =============== Allocation and MAE computation =============== #

# N_days
if year%4 == 0:
    N_days = 366
else:
    N_days = 365
    
# others
N_fcst = 54
EN = 75
N_grids = BCH_obs.shape[-1]

# ---------- allocations ---------- #
MAE = np.empty((N_days, N_fcst, N_grids))
SPREAD = np.empty((N_days, N_fcst, N_grids))
CRPS = np.empty((N_days, N_fcst, N_grids))

print("Computing CRPS ...")

for lead in range(N_fcst):
    print("lead = {}".format(lead))
    
    with h5py.File(REFCST_dir + "{}_{}_lead{}.hdf".format(perfix_raw, year, lead), 'r') as h5io:
        AnEn = h5io[key_raw][:, :EN, ...][..., indx, indy]
        
    crps, mae, _ = metrics.CRPS_1d_nan(BCH_obs[:, lead, :], AnEn)
    MAE[:, lead, ...] = mae
    CRPS[:, lead, ...] = crps

tuple_save = (MAE, CRPS,)
label_save = ['MAE', 'CRPS',]
du.save_hdf5(tuple_save, label_save, save_dir, '{}_CRPS_RAW_{}.hdf'.format(perfix_raw, year))
