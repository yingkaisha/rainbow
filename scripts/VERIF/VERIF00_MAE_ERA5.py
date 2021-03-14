'''
Computing the mean absolute error (MAE) of BCH obs and ERA5 values at stn grids.
'''

import sys
import argparse
from glob import glob
from datetime import datetime, timedelta

# data tools
import h5py
import time
import numpy as np

# custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/')
sys.path.insert(0, '/glade/u/home/ksha/PUBLISH/fcstpp/')

from fcstpp import metrics
import data_utils as du
from namelist import * 

# ---------- Parsers ---------- #
parser = argparse.ArgumentParser()
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

year = int(args['year'])

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
    
# other params
N_fcst = 54; EN = 75
N_grids = BCH_obs.shape[-1]

MAE = np.empty((N_days, N_fcst, N_grids))

print("Computing MAE ...")

for lead in range(N_fcst):
    print("lead = {}".format(lead))
    
    with h5py.File(ERA_dir+'ERA5_GEFS-fcst_{}.hdf'.format(year), 'r') as h5io:
        ERA_true = h5io['era_fcst'][..., lead, bc_inds[0]:bc_inds[1], bc_inds[2]:bc_inds[3]]
    ERA_true = ERA_true[..., indx, indy]    
    MAE[:, lead, ...] = np.abs(BCH_obs[:, lead, :]-ERA_true)

tuple_save = (MAE,)
label_save = ['MAE',]
du.save_hdf5(tuple_save, label_save, save_dir, 'ERA5_MAE_BCH_{}.hdf'.format(year))


