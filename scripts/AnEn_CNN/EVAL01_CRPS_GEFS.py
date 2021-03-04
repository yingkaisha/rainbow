
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

# ---------- Parsers ---------- #
parser = argparse.ArgumentParser()
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

year = int(args['year'])

# ---------- BCH ---------- #
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

with h5py.File(save_dir+'BCH_ERA5_GEFS_pairs.hdf', 'r') as h5io:
    BCH_obs = h5io['BCH_obs'][...]
    indx = h5io['indx'][...]
    indy = h5io['indy'][...]
    
BCH_obs = BCH_obs[flag_pick, ...]

# ---------- params ---------- #
# N_days
if year%4 == 0:
    N_days = 366
else:
    N_days = 365
    
# others
N_fcst = 54
EN = 45
N_grids = BCH_obs.shape[-1]

# ---------- allocations ---------- #
MAE = np.empty((N_days, N_fcst, N_grids))
SPREAD = np.empty((N_days, N_fcst, N_grids))
CRPS = np.empty((N_days, N_fcst, N_grids))

print("Computing CRPS ...")

for lead in range(N_fcst):
    print("lead = {}".format(lead))
    
    with h5py.File(REFCST_dir + "GEFS_QM_{}_lead{}.hdf".format(year, lead), 'r') as h5io:
        GEFS = h5io['gefs_qm'][...][..., indx, indy]
        
    crps, mae, _ = metrics.CRPS_1d_nan(BCH_obs[:, lead, :], GEFS)
    
    MAE[:, lead, ...] = mae
    CRPS[:, lead, ...] = crps

tuple_save = (MAE, CRPS,)
label_save = ['MAE', 'CRPS',]
du.save_hdf5(tuple_save, label_save, save_dir, 'GEFS_CRPS_BCH_{}.hdf'.format(year))
