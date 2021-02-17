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
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

year = int(args['year'])

# ---------- params ---------- #
# N_days
if year%4 == 0:
    N_days = 366
else:
    N_days = 365
    
# others
N_fcst = 54
EN = 75

# ---------- grided info ---------- #
with h5py.File(save_dir+'BC_domain_info.hdf', 'r') as h5io:
    land_mask_bc = h5io['land_mask_bc'][...]
    
grid_shape = land_mask_bc.shape

# allocations
MAE = np.empty((N_days, N_fcst,)+grid_shape)
SPREAD = np.empty((N_days, N_fcst,)+grid_shape)
CRPS = np.empty((N_days, N_fcst,)+grid_shape)

print("Computing CRPS ...")

for lead in range(N_fcst):
    print("lead = {}".format(lead))
    
    with h5py.File(ERA_dir+'ERA5_GEFS-fcst_{}.hdf'.format(year), 'r') as h5io:
        ERA_true = h5io['era_fcst'][..., lead, bc_inds[0]:bc_inds[1], bc_inds[2]:bc_inds[3]]
    
    with h5py.File(REFCST_dir + "GEFS_RAW_{}_lead{}.hdf".format(year, lead), 'r') as h5io:
        GEFS = h5io['gefs_qm'][...]
        
    crps, mae, _ = metrics.CRPS_2d(ERA_true, GEFS, ~land_mask_bc)
    
    MAE[:, lead, ...] = mae
    CRPS[:, lead, ...] = crps
    print("... Done")

tuple_save = (MAE, CRPS,)
label_save = ['MAE', 'CRPS',]
du.save_hdf5(tuple_save, label_save, save_dir, 'GEFS_RAW_CRPS_{}.hdf'.format(year))