
import sys
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
sys.path.insert(0, '/glade/u/home/ksha/PUBLISH/fcstpp/')

from fcstpp import gridpp
import analog_utils as ana
import data_utils as du
from namelist import * 

lead0 = 0
lead1 = 54

@nb.njit()
def simple_max(a, b):
    if a >= b:
        return a
    else:
        return b

@nb.njit()
def simple_min(a, b):
    if a <= b:
        return a
    else:
        return b

@nb.njit()
def quantile_mapping_stencil(pred, cdf_pred, cdf_true, land_mask, rad=1):
    '''
    pred = (en, grids)
    cdf = (quantile, grids)
    '''
    EN, Nx, Ny = pred.shape
    N_fold = (2*rad+1)**2
    out = np.empty((EN, N_fold, Nx, Ny,))
    out[...] = np.nan
    
    slope = 0
    for i in range(Nx):
        for j in range(Ny):
            if land_mask[i, j]:
                
                # handling edging grid points
                min_x = simple_max(i-rad, 0)
                max_x = simple_min(i+rad, Nx-1)
                min_y = simple_max(j-rad, 0)
                max_y = simple_min(j+rad, Ny-1)
                
                # counting stencil grids
                count = 0
                
                # center grid = (i, j); stencil grids = (ix, iy)
                for ix in range(min_x, max_x+1):
                    for iy in range(min_y, max_y+1):
                        if land_mask[ix, iy]:
                            for en in range(EN):
                                #out[en, count, i, j] = np.interp(pred[en, ix, iy], cdf_pred[:, ix, iy], cdf_true[:, i, j])                        
                                if pred[en, ix, iy] <= cdf_pred[93, ix, iy]:
                                    out[en, count, i, j] = np.interp(pred[en, ix, iy], cdf_pred[:, ix, iy], cdf_true[:, i, j])                                
                                else:
                                    slope = np.sum((cdf_true[93:, i, j] - cdf_true[93, i, j])*(cdf_pred[93:, ix, iy] - cdf_pred[93, ix, iy]))/\
                                                   np.sum((cdf_pred[93:, ix, iy] - cdf_pred[93, ix, iy])**2)
                                    out[en, count, i, j] = cdf_true[93, i, j] + slope*(pred[en, ix, iy] - cdf_pred[93, ix, iy])
                                    
                            count += 1
    return out

@nb.njit()
def enlarge_stencil(pred, land_mask, rad=1):
    '''
    pred = (en, grids)
    cdf = (quantile, grids)
    '''
    EN, Nx, Ny = pred.shape
    N_fold = (2*rad+1)**2
    out = np.empty((EN, N_fold, Nx, Ny,))
    out[...] = np.nan
    
    slope = 0
    for i in range(Nx):
        for j in range(Ny):
            if land_mask[i, j]:
                
                # handling edging grid points
                min_x = simple_max(i-rad, 0)
                max_x = simple_min(i+rad, Nx-1)
                min_y = simple_max(j-rad, 0)
                max_y = simple_min(j+rad, Ny-1)
                
                # counting stencil grids
                count = 0
                
                # center grid = (i, j); stencil grids = (ix, iy)
                for ix in range(min_x, max_x+1):
                    for iy in range(min_y, max_y+1):
                        if land_mask[ix, iy]:
                            for en in range(EN):
                                out[en, count, i, j] = np.interp(pred[en, ix, iy], cdf_pred[:, ix, iy], cdf_true[:, i, j])                        
                            count += 1
    return out  
    

with h5py.File(save_dir+'BC_domain_info.hdf', 'r') as h5io:
    land_mask_bc = h5io['land_mask_bc'][...]

# ERA quantiles
CDF_era = np.empty((12, 105, 48, 112)); CDF_era[...] = np.nan
with h5py.File(ERA_dir+'PT_3hour_quantile.hdf', 'r') as h5io:
    CDF_era[..., ~land_mask_bc] = h5io['CDF'][...]
    q_bins = h5io['q'][...]

# output allocation
EN = 45
gefs_out = np.empty((366, 5, 9, 48, 112))
flag_pick = np.zeros((366,), dtype=int)

freq = 3.0
FCSTs = np.arange(9, 240+freq, freq) # fcst lead as hour

for year in range(2017, 2020):
    
    print('year: {}'.format(year))
    
    if year%4 == 0:
        N_days = 366
    else:
        N_days = 365
        
    base = datetime(year, 1, 1)
    date_list = [base + timedelta(days=x) for x in range(N_days)]

    # loop over lead times
    for lead in range(lead0, lead1):
        for d, date in enumerate(date_list):
            
            # ini date + lead time
            date_true = date + timedelta(hours=FCSTs[lead])
            flag_pick[d] = date_true.month-1
            
        print("lead = {}".format(lead))
        
        # fcst quantiles
        CDF_fcst = np.empty((12, 105, 48, 112)); CDF_fcst[...] = np.nan
        with h5py.File(save_dir+'GEFS_lead{}_quantile.hdf'.format(lead), 'r') as h5io:
            CDF_fcst[..., ~land_mask_bc] = h5io['CDF'][...]
        
        # raw fcst
        with h5py.File(REFCST_dir+'En_members_APCP_{}.hdf'.format(year), 'r') as h5io:
            gefs = h5io['bc_mean'][:, lead, ...]

        for i in range(N_days):
            ind_ = flag_pick[i]
            gefs_out[i, ...] = quantile_mapping_stencil(gefs[i, ...], CDF_fcst[ind_, ...], CDF_era[ind_, ...], ~land_mask_bc)
            
        gefs_save = np.reshape(gefs_out[:N_days, ...], (N_days,)+(45, 48, 112))
        gefs_save = gridpp.bootstrap_fill(gefs_save, EN, ~land_mask_bc)
        
        tuple_save = (gefs_save,)
        label_save = ['gefs_qm']
        du.save_hdf5(tuple_save, label_save, REFCST_dir, 'GEFS_QM_{}_lead{}.hdf'.format(year, lead))
