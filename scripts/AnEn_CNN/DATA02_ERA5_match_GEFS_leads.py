import sys
from glob import glob
from datetime import datetime, timedelta

# data tools
import h5py
import numpy as np
#import netCDF4 as nc

# custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/')
import data_utils as du
from namelist import * 

N_fcst = 78
freq = 3.0
FCSTs = np.arange(9, 240+freq, freq) # fcst lead as hours

print("Importing ERA5 reanalysis")
PCT_history = {}
for year in range(2000, 2021):
    with h5py.File(ERA_dir+'PT_3hr_{}.hdf'.format(year), 'r') as h5io:
        era_pct = h5io['era_025'][...]
    PCT_history['{}'.format(year)] = era_pct

for year in range(2000, 2020):
    print("Processing year: {}".format(year))
    N_days = (datetime(year+1, 1, 1)-datetime(year, 1, 1)).days
    ERA_fcst = np.zeros((N_days, N_fcst, 160, 220))
    
    for day in range(N_days):
        for t, fcst_temp in enumerate(FCSTs):
            # fcsted (targeted) date
            date_true = datetime(year, 1, 1)+timedelta(days=day)+timedelta(hours=fcst_temp)
            # handling cross years 
            year_true = int(date_true.year)
            ind_true = int((date_true-datetime(year_true, 1, 1)).total_seconds()/60/60/freq)
            # 
            ERA_fcst[day, t, ...] = PCT_history['{}'.format(year_true)][ind_true, ...]
            
    ERA_fcst[ERA_fcst<1e-5] = 0.0
    tuple_save = (ERA_fcst,)
    label_save = ['era_fcst']

    filename = 'ERA5_GEFS-fcst_{}.hdf'.format(year)
        
    du.save_hdf5(tuple_save, label_save, ERA_dir, filename) 

