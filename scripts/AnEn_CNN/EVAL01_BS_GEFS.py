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

def quantile_to_flag(data, quantile_grids, year, period=3):
    '''
    determine if gridded fcst/obs is larger than 
    the given grid-point-wise quantile values.  
    '''
    
    input_shape = data.shape
    output = np.empty(input_shape)
    output[...] = np.nan
    
    if year%4 == 0:
        N_days = 366
    else:
        N_days = 365
    
    base = datetime(year, 1, 1)
    date_list = [base + timedelta(hours=x) for x in range(0, N_days*24, period)]
    
    # no ensemble dimensions (time, grid)
    if len(input_shape) == 2:    
        for d, date in enumerate(date_list):
            mon_ind = date.month-1
            output[d, :] = data[d, :] > quantile_grids[mon_ind, :]
            
    # (time, ensemble, grid)
    else:
        for d, date in enumerate(date_list):
            mon_ind = date.month-1
            for en in range(input_shape[1]):
                output[d, en, :] = data[d, en, :] > quantile_grids[mon_ind, :]
    return output

parser = argparse.ArgumentParser()
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

year_target = int(args['year'])

# ---------- params ---------- #
# N_days
if year_target%4 == 0:
    N_days = 366
else:
    N_days = 365
    
# others
N_fcst = 54
EN = 45
# ---------- grided info ---------- #
with h5py.File(save_dir+'BC_domain_info.hdf', 'r') as h5io:
    land_mask_bc = h5io['land_mask_bc'][...]

# ERA quantiles
with h5py.File(ERA_dir+'PT_3hour_quantile.hdf', 'r') as h5io:
    CDF_era = h5io['CDF'][...]
    q_bins = h5io['q'][...]
    
# allocations
BS_FCST_q90 = np.empty((N_days, N_fcst,))
BS_FCST_mm = np.empty((N_days, N_fcst, 2))

for lead in range(N_fcst):
    
    print("Brier Score for lead: {}".format(lead))
    
    with h5py.File(ERA_dir+'ERA5_GEFS-fcst_{}.hdf'.format(year_target), 'r') as h5io:
        era_fcst = h5io['era_fcst'][..., lead, bc_inds[0]:bc_inds[1], bc_inds[2]:bc_inds[3]]
    era_fcst = era_fcst[..., ~land_mask_bc]
    
    with h5py.File(REFCST_dir + "GEFS_QM_{}_lead{}.hdf".format(year_target, lead), 'r') as h5io:
        GEFS = h5io['gefs_qm'][...]
    GEFS = GEFS[..., ~land_mask_bc]
    
    for p in np.array([0.9,]):

        quantile_grids = CDF_era[:, :-1, :]
        quantile_grids = quantile_grids[:, q_bins==p, :]

        # converting each historical years into flags
        # these flags will be averaged as historical probabilities
        
        era_cate = quantile_to_flag(era_fcst, quantile_grids, year=year_target, period=24)
        fcst = quantile_to_flag(GEFS, quantile_grids, year=year_target, period=24)

        BS_FCST_q90[:, lead] = np.sum(metrics.BS_binary_1d(era_cate, fcst), axis=1)
        
    for i, THRES in enumerate(np.array([5, 30.0])):
    
        thres = THRES/8.0
        era_cate = era_fcst > thres
        fcst = GEFS > thres
        
        BS_FCST_mm[:, lead, i] = np.sum(metrics.BS_binary_1d(era_cate, fcst), axis=1)

tuple_save = (BS_FCST_q90,)
label_save = ['BS']
du.save_hdf5(tuple_save, label_save, save_dir, 'GEFS_BS-90th_{}.hdf'.format(year_target))

tuple_save = (BS_FCST_mm[..., 0],)
label_save = ['BS']
du.save_hdf5(tuple_save, label_save, save_dir, 'GEFS_BS-5mm_{}.hdf'.format(year_target))

tuple_save = (BS_FCST_mm[..., 1],)
label_save = ['BS']
du.save_hdf5(tuple_save, label_save, save_dir, 'GEFS_BS-30mm_{}.hdf'.format(year_target))

