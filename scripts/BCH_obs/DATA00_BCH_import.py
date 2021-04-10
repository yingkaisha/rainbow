'''
Processing BC Hydro raw bucket height observation 
files as hourly and 3 hourly resample precipitation rates.
'''

import sys
from glob import glob

import h5py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/QC_OBS/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/QC_OBS/utils')

import data_utils as du
import BCH_utils as bu
from namelist import *


filenames = sorted(glob(BACKUP_dir+'BCH_2016_2020/*.txt'))

keys, keys_stored = bu.BCH_txt_preprocess(filenames, BACKUP_dir+'BCH_PREC_QC_NRT_2016_2020.hdf', 
                                          ['PREC_INST_QC',], qc_code=['50',], verbose=False)

date_start = datetime(2016, 1, 1, 0, 0, 0)
date_end = datetime(2021, 1, 1, 0, 0, 0)

with pd.HDFStore(BACKUP_dir+'BCH_PREC_QC_NRT_2016_2020.hdf', 'r') as hdf_io:
    keys = hdf_io.keys()
keys = du.del_slash(keys)

# ========== hourly ========== #

with pd.HDFStore(BACKUP_dir+'BCH_PREC_QC_hourly_2016_2020.hdf', 'w') as hdf_io:
    for key in keys:
        with pd.HDFStore(BACKUP_dir+'BCH_PREC_QC_NRT_2016_2020.hdf', 'r') as hdf_nrt_io:
            BCH_pd = hdf_nrt_io[key]
            
        # PST to UTC
        BCH_pd['datetime'] = BCH_pd['datetime'].dt.tz_localize('America/Los_Angeles', ambiguous='NaT', nonexistent='NaT')
        BCH_pd['datetime'] = BCH_pd['datetime'].dt.tz_convert('GMT')
        BCH_pd = BCH_pd.dropna()
        
        bucket_height = BCH_pd['PREC_INST_QC'].values
        sec_obs = (BCH_pd['datetime'].values.astype('O')/1e9).astype(int)
        
        precip, date_ref = bu.BCH_PREC_resample(bucket_height, sec_obs, date_start, date_end, period=60*60)
        
        # create a new dataframe as the output
        temp_pd2 = pd.DataFrame()
        
        # assigning datetime and vals
        temp_pd2['datetime'] = date_ref
        temp_pd2['PREC_HOUR_QC'] = precip
        
        hdf_io[key] = temp_pd2

# ========== 3 hourly ========== #

with pd.HDFStore(BACKUP_dir+'BCH_PREC_QC_3H_2016_2020.hdf', 'w') as hdf_io:
    for key in keys:
        with pd.HDFStore(BACKUP_dir+'BCH_PREC_QC_NRT_2016_2020.hdf', 'r') as hdf_nrt_io:
            BCH_pd = hdf_nrt_io[key]
        
        # PST to UTC
        BCH_pd['datetime'] = BCH_pd['datetime'].dt.tz_localize('America/Los_Angeles', ambiguous='NaT', nonexistent='NaT')
        BCH_pd['datetime'] = BCH_pd['datetime'].dt.tz_convert('GMT')
        BCH_pd = BCH_pd.dropna()
        
        bucket_height = BCH_pd['PREC_INST_QC'].values
        sec_obs = (BCH_pd['datetime'].values.astype('O')/1e9).astype(int)

        precip, date_ref = bu.BCH_PREC_resample(bucket_height, sec_obs, date_start, date_end, period=3*60*60)
        
        # create a new dataframe as the output
        temp_pd2 = pd.DataFrame()
        
        # assigning datetime and vals
        temp_pd2['datetime'] = date_ref
        temp_pd2['PREC_HOUR_QC'] = precip

        hdf_io[key] = temp_pd2
