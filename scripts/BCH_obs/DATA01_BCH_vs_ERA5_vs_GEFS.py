
import sys
from glob import glob

import h5py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/QC_OBS/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/QC_OBS/utils')

import graph_utils as gu
import data_utils as du
import BCH_utils as bu
from namelist import *

years = [2017, 2018, 2019]

with h5py.File(save_dir+'BCH_ERA5_GEFS_pairs.hdf', 'r') as h5io:
    ERA5_obs = h5io['ERA5_obs'][...]
    GEFS_obs = h5io['GEFS_obs'][...]
    BCH_obs = h5io['BCH_obs'][...]

N_fcst = 54
EN = 75
N_grids = BCH_obs.shape[-1]

with h5py.File(save_dir+'BCH_ERA5_GEFS_pairs.hdf', 'r') as h5io:
    indx = h5io['indx'][...]
    indy = h5io['indy'][...]

with h5py.File(save_dir+'NA_SL_info.hdf', 'r') as h5io:
    W_SL = h5io['W_SL'][bc_inds[0]:bc_inds[1], bc_inds[2]:bc_inds[3]][indx, indy]

filename_smooth = [REFCST_dir+"BASE_final_{}_lead{}.hdf",
                   REFCST_dir+"SL_final_{}_lead{}.hdf",
                   REFCST_dir+"BASE_CNN_QM_{}_lead{}.hdf",
                   REFCST_dir+"SL_CNN_QM_{}_lead{}.hdf",]

filename_raw = [REFCST_dir+"BASE_final_{}_lead{}.hdf",
                REFCST_dir+"SL_final_{}_lead{}.hdf",
                REFCST_dir+"BASE_final_{}_lead{}.hdf",
                REFCST_dir+"SL_final_{}_lead{}.hdf", 
                REFCST_dir+"GEFS_QM_{}_lead{}.hdf",
                ERA_dir+'ERA5_GEFS-fcst_{}.hdf']

key_smooth_list = ['AnEn_SG', 'AnEn_SG', 'cnn_pred', 'cnn_pred']
key_raw_list = ['AnEn',]*4 + ['gefs_qm', 'era_fcst']

L = 365 + 365 + 365
AnEn_BCH = np.empty((L, N_fcst, N_grids))

tuple_save = ()
label_save = ['BASE_final', 'SL_final', 'BASE_CNN', 'SL_CNN', 'GEFS', 'ERA5']

# 4 post-processed, quantile mapped GEFS, ERA5

for i in range(6):
    
    print("Extracting {}".format(label_save[i]))
    
    # ERA5 
    if i == 5:
        AnEn_mean = ()
        
        for year in years:
            print(filename_raw[i].format(year, lead))
            
            with h5py.File(filename_raw[i].format(year), 'r') as h5io:
                era5_ = h5io[key_raw_list[i]][..., :N_fcst, bc_inds[0]:bc_inds[1], bc_inds[2]:bc_inds[3]][..., indx, indy]                
            pred_ = era5_
            
            AnEn_mean += (pred_,)
            
        AnEn_BCH_ = np.concatenate(AnEn_mean, axis=0)
        
        print('store all data')
        tuple_save += (np.copy(AnEn_BCH_),)
        
    # GEFS and post-processed GEFS     
    else:
        for lead in range(N_fcst):

            AnEn_mean = ()

            for year in years:

                if i < 4:
                    print(filename_raw[i].format(year, lead))

                    with h5py.File(filename_raw[i].format(year, lead), 'r') as h5io:
                        RAW = h5io[key_raw_list[i]][:, :EN, ...][..., indx, indy]

                    with h5py.File(filename_smooth[i].format(year, lead), 'r') as h5io:
                        SMOOTH = h5io[key_smooth_list[i]][:, :EN, ...][..., indx, indy]

                    AnEn = W_SL*RAW + (1-W_SL)*SMOOTH
                    pred_ = np.mean(AnEn, axis=1)

                if i == 4:
                    print(filename_raw[i].format(year, lead))

                    with h5py.File(filename_raw[i].format(year, lead), 'r') as h5io:
                        gefs_ = h5io[key_raw_list[i]][:, :EN, ...][..., indx, indy]
                    pred_ = np.mean(gefs_, axis=1)
                    
                AnEn_mean += (pred_,)

            print('store lead time')
            AnEn_BCH[:, lead, :] = np.concatenate(AnEn_mean, axis=0)
            
        print('store all data')
        tuple_save += (np.copy(AnEn_BCH),)


du.save_hdf5(tuple_save, label_save, save_dir, 'BCH_MODEL_pairs.hdf')


