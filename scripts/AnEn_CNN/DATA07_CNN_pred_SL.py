# general tools
import sys
import h5py
import argparse

# data tools
import time
from datetime import datetime, timedelta
import numpy as np

from tensorflow import keras
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

# custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/utils/')

from namelist import *
import data_utils as du
import model_utils as mu
import train_utils as tu

parser = argparse.ArgumentParser()
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

# =============== #
year = int(args['year'])
print('Pred on {}'.format(year))

if year%4 == 0:
    N_day = 366
else:
    N_day = 365    

datelist = [datetime(year, 1, 1, 0)+timedelta(days=i) for i in range(N_day)]
L = len(datelist)

# =============== #
LEAD = 54
TYPE = 'SL'
EN = 75
freq = 3.0
FCSTs = np.arange(9, 240+freq, freq) # fcst lead as hours

# =============== #
input_tensor = Input((None, None, 3))
filter_num_down = [80, 160, 320, 640]
filter_num_skip = [80, 80, 80,]
filter_num_aggregate = 320
filter_num_sup = 80
stack_num_down = 2
stack_num_up = 1
activation = 'GELU'
batch_norm = True
pool = False
unpool = False
name = 'denoise'

X_decoder = mu.denoise_base(input_tensor, filter_num_down, filter_num_skip, filter_num_aggregate, 
                            stack_num_down=stack_num_down, stack_num_up=stack_num_up, activation=activation, 
                            batch_norm=batch_norm, pool=pool, unpool=unpool, name=name)

OUT_stack = mu.denoise_sup_head(X_decoder, filter_num_sup, activation=activation, 
                                batch_norm=batch_norm, pool=pool, unpool=unpool, name=name)

model = Model([input_tensor,], OUT_stack)

model.compile(loss=[keras.losses.mean_absolute_error, 
                    keras.losses.mean_absolute_error,
                    keras.losses.mean_absolute_error,], 
              loss_weights=[0.1, 0.1, 1.0],
              optimizer=keras.optimizers.SGD(lr=5e-6))

model_path = temp_dir + 'AnEn_UNET3M_RAW_tune.hdf'
W = mu.dummy_loader(model_path)

model.set_weights(W)

# =============== #
# elev and land mask
with h5py.File(save_dir+'BC_domain_info.hdf', 'r') as h5io:
    etopo_025 = h5io['etopo_bc'][...]
    land_mask_bc = h5io['land_mask_bc'][...]
grid_shape = land_mask_bc.shape

# clim
with h5py.File(ERA_dir+'PT_3hour_clim.hdf', 'r') as h5io:
    era_3h_clim = h5io['era_3h_clim'][..., bc_inds[0]:bc_inds[1], bc_inds[2]:bc_inds[3]]

# precip preprocess
era_3h_clim = np.log(era_3h_clim+1)
era_3h_clim[..., land_mask_bc] = 0.0

# elevation preprocess
etopo_025[etopo_025<0] = 0
max_ = np.nanmax(etopo_025)
min_ = np.nanmin(etopo_025)
etopo_025 = (etopo_025-min_)/(max_-min_)

# allocations
N_sample = N_day
cgan_raw = np.empty((N_sample, EN)+grid_shape)
RAW = np.empty((N_sample, EN)+grid_shape); RAW[...] = np.nan
CLIM_elev = np.empty((L, EN)+grid_shape+(2,))
mon_inds = np.empty((L,), dtype=int)

# =============== #
# loop over lead times
for lead in range(LEAD):
    print('lead = {}'.format(lead))
    
    # identify the month of (ini time + lead time)
    for i, date in enumerate(datelist):
        date_true = date + timedelta(hours=FCSTs[lead])
        mon_inds[i] = date_true.month-1
    
    # preparing CNN inputs
    for i in range(L):
        for en in range(EN):
            CLIM_elev[i, en, ..., 0] = era_3h_clim[mon_inds[i], ...]
            CLIM_elev[i, en, ..., 1] = etopo_025

    with h5py.File(REFCST_dir + "{}_final_dress_SS_{}_lead{}.hdf".format(TYPE, year, lead), 'r') as h5io:
        H15_SL = h5io['AnEn'][:N_sample, :EN, ...]
        
    # noisy AnEn preprocess
    H15_SL[H15_SL<0]=0
    H15_SL = np.log(H15_SL+1)
    RAW[..., ~land_mask_bc] = H15_SL
    RAW[..., land_mask_bc] = 0
    
    for i in range(N_sample):    
        X = np.concatenate((RAW[i, ..., None], CLIM_elev[i, ...]), axis=-1)
        temp_ = model.predict([X])
        cgan_raw[i, ...] = temp_[-1][..., 0]
    
    cgan_raw[cgan_raw<0] = 0
    cgan_raw = np.exp(cgan_raw)-1 # <--- de-normalized
    
    cgan_raw[..., land_mask_bc] = np.nan
    
    tuple_save = (cgan_raw,)
    label_save = ['cnn_pred',]
    du.save_hdf5(tuple_save, label_save, REFCST_dir, '{}_CNN_{}_lead{}.hdf'.format(TYPE, year, lead))
    