
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
    perfix_smooth = 'BASE_final'
    perfix_raw = 'BASE_final'
    key_smooth = 'AnEn_SG'
    key_raw = 'AnEn'
    
elif type_ind == 1:
    perfix_smooth = 'SL_final'
    perfix_raw = 'SL_final'
    key_smooth = 'AnEn_SG'
    key_raw = 'AnEn'
    
elif type_ind == 2:
    perfix_smooth = 'BASE_CNN'
    perfix_raw = 'BASE_final'
    key_smooth = 'cnn_pred'
    key_raw = 'AnEn'
    
elif type_ind == 3:
    perfix_smooth = 'SL_CNN'
    perfix_raw = 'SL_final'
    key_smooth = 'cnn_pred'
    key_raw = 'AnEn'

print("perfix_smooth = {}; perfix_raw = {}".format(perfix_smooth, perfix_raw))

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
with h5py.File(save_dir+'NA_SL_info.hdf', 'r') as h5io:
    W_SL = h5io['W_SL'][bc_inds[0]:bc_inds[1], bc_inds[2]:bc_inds[3]]

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
    
    with h5py.File(REFCST_dir + "{}_{}_lead{}.hdf".format(perfix_raw, year, lead), 'r') as h5io:
        RAW = h5io[key_raw][:, :EN, ...]
            
    with h5py.File(REFCST_dir + "{}_{}_lead{}.hdf".format(perfix_smooth, year, lead), 'r') as h5io:
        SMOOTH = h5io[key_smooth][:, :EN, ...]
    
    AnEn = W_SL*RAW + (1-W_SL)*SMOOTH
    
    crps, mae, _ = metrics.CRPS_2d(ERA_true, AnEn, ~land_mask_bc)
    MAE[:, lead, ...] = mae
    CRPS[:, lead, ...] = crps
    print("... Done")

tuple_save = (MAE, CRPS,)
label_save = ['MAE', 'CRPS',]
du.save_hdf5(tuple_save, label_save, save_dir, '{}_CRPS_{}.hdf'.format(perfix_smooth, year))
