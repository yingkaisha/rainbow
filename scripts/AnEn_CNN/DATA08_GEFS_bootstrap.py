
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


with h5py.File(save_dir+'BC_domain_info.hdf', 'r') as h5io:
    land_mask_bc = h5io['land_mask_bc'][...]

for year in range(2017, 2020):
    
    print('year: {}'.format(year))
    
    if year%4 == 0:
        N_days = 366
    else:
        N_days = 365

    # loop over lead times
    for lead in range(lead0, lead1):
        print("lead = {}".format(lead))

        # raw fcst
        with h5py.File(REFCST_dir+'En_members_APCP_{}.hdf'.format(year), 'r') as h5io:
            gefs = h5io['bc_mean'][:, lead, ...]
            
        # bootstrap
        gefs_save = gridpp.bootstrap_fill(gefs, 75, ~land_mask_bc)
        
        tuple_save = (gefs_save,)
        label_save = ['gefs_qm']
        du.save_hdf5(tuple_save, label_save, REFCST_dir, 'GEFS_RAW_{}_lead{}.hdf'.format(year, lead))
