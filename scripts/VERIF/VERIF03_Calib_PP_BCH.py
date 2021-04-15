import sys
import argparse
from glob import glob
from datetime import datetime, timedelta

# data tools
import h5py
import pandas as pd
import numba as nb
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/')
sys.path.insert(0, '/glade/u/home/ksha/PUBLISH/fcstpp/')

from sklearn.metrics import brier_score_loss
from fcstpp import metrics, utils
import analog_utils as ana
import data_utils as du
from namelist import * 

def reliability_diagram(cate_true, prob_model, bins):
    binids = np.searchsorted(bins, prob_model)
    bin_sums = np.bincount(binids, weights=prob_model, minlength=len(bins))
    bin_true = np.bincount(binids, weights=cate_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))
    flag = bin_total > 0
    prob_true = bin_true/bin_total
    prob_pred = bin_sums/bin_total
    prob_true[~flag] = np.nan
    return prob_true, prob_pred

def fcst_to_prob(data, thres, mon_inds):
    dim1, dim2, dim3, dim4 = data.shape
    
    out = np.empty((dim1, dim2, dim3, dim4))
    
    for i in range(dim1):
        thres_ = thres[mon_inds[i], :]
        for j in range(dim2):
            for k in range(dim3):
                out[i, j, k, :] = data[i, j, k, :] > thres_
    return np.sum(out, axis=2)/dim3

def fcst_to_flag(data, thres, mon_inds):
    dim1, dim2, dim3 = data.shape
    out = np.empty((dim1, dim2, dim3))
    
    for i in range(dim1):
        thres_ = thres[mon_inds[i], :]
        for j in range(dim2):
            flag_nan = np.isnan(data[i, j, :])
            out[i, j, :] = data[i, j, :] > thres_
            out[i, j, flag_nan] = np.nan
    return out

# ---------- Parsers ---------- #
parser = argparse.ArgumentParser()
parser.add_argument('out', help='out')
args = vars(parser.parse_args())

type_ind = int(args['out'])

if type_ind == 0:
    prefix_raw = 'BASE_final_SS'
    prefix_out = 'BASE_final'
    key_raw = 'AnEn'
    EN = 25
    
elif type_ind == 1:
    prefix_raw = 'SL_final_SS'
    prefix_out = 'SL_final'
    key_raw = 'AnEn'
    EN = 25
    
elif type_ind == 2:
    prefix_raw = 'BASE_CNN'
    prefix_out = 'BASE_CNN'
    key_raw = 'cnn_pred'
    EN = 75 # 25 members dressed to 75
    
elif type_ind == 3:
    prefix_raw = 'SL_CNN'
    prefix_out = 'SL_CNN'
    key_raw = 'cnn_pred'
    EN = 75 # 25 members dressed to 75

# ========== Gridded input ========== #

# three watershed groups
with h5py.File(save_dir+'BCH_wshed_groups.hdf', 'r') as h5io:
    flag_sw = h5io['flag_sw'][...]
    flag_si = h5io['flag_si'][...]
    flag_n = h5io['flag_n'][...]
FLAGs = (flag_sw, flag_si, flag_n)

with h5py.File(save_dir+'BC_domain_info.hdf', 'r') as h5io:
    land_mask_bc = h5io['land_mask_bc'][...]
grid_shape = land_mask_bc.shape
    
# ========== BCH obs preprocessing ========== # 

# import station obsevations and grid point indices
with h5py.File(save_dir+'BCH_ERA5_3H_verif.hdf', 'r') as h5io:
    BCH_obs = h5io['BCH_obs'][...]
    indx = h5io['indx'][...]
    indy = h5io['indy'][...]
    
# subsetting BCH obs into a given year
N_days_bch = 366 + 365*3
date_base_bch = datetime(2016, 1, 1)
date_list_bch = [date_base_bch + timedelta(days=x) for x in np.arange(N_days_bch, dtype=np.float)]

flag_pick = []
for date in date_list_bch:
    if date.year in [2017, 2018, 2019]:
        flag_pick.append(True)
    else:
        flag_pick.append(False)

flag_pick = np.array(flag_pick)

BCH_obs = BCH_obs[flag_pick, ...]

# number of stations
N_stn = BCH_obs.shape[-1]

# ========== ERA5 stn climatology preprocessing ========== #

# importing girdded ERA5 quantiles
with h5py.File(ERA_dir+'PT_3hour_q.hdf', 'r') as h5io:
    CDF_era = h5io['era_3hq_bc'][...]
    q_bins = h5io['q_bins'][...]

CDF_obs = np.empty((12, 107,)+land_mask_bc.shape)
CDF_obs[..., ~land_mask_bc] = CDF_era
CDF_obs = CDF_obs[..., indx, indy]

# station and monthly (contains neighbouring months) wise 90th
BCH_90th = CDF_obs[:, 93, :] 

# ========== Merging multi-year post-processed files ========== #

# datelist for 2017-2019
N_days = 365 + 365 + 365
date_base = datetime(2017, 1, 1)
date_list = [date_base + timedelta(days=x) for x in np.arange(N_days, dtype=np.float)]

# allocation
AnEn_stn = np.empty((N_days, N_fcst, EN, N_stn))

for y in [2017, 2018, 2019]:
    if y%4 == 0:
        n_days = 366
    else:
        n_days = 365
        
    if type_ind == 0 or type_ind == 1:
        AnEn_full = np.empty((n_days, EN)+grid_shape)
        AnEn_full[...] = np.nan
    
    # ---------- #
    flag_pick = []
    for date in date_list:
        if date.year == y:
            flag_pick.append(True)
        else:
            flag_pick.append(False)
    flag_pick = np.array(flag_pick)
    # ---------- #
    
    for lead in range(N_fcst):
        
        with h5py.File(REFCST_dir + "{}_{}_lead{}.hdf".format(prefix_raw, y, lead), 'r') as h5io:
            AnEn_ = h5io[key_raw][:, :EN, ...]
        
        if type_ind == 0 or type_ind == 1:
            # id 0 and 1 are flattened grid points, reshape them to 2d.
            AnEn_full[..., ~land_mask_bc] = AnEn_
            AnEn_stn[flag_pick, lead, ...] = AnEn_full[..., indx, indy]
        else:
            # cnn outputs can be negative, fix it here.
            AnEn_stn[flag_pick, lead, ...] = ana.cnn_precip_fix(AnEn_[..., indx, indy])

# ========== Calibration ========== #
# params
N_bins = 15
N_boost = 100
hist_bins = np.linspace(0, 1, N_bins)
N_lead_day = 7 # number of forecasted days

# 3-hr lead times to days
fcst_leads_ini = np.arange(0, 72*3+3, 3, dtype=np.float)
# date_base = datetime(2017, 1, 1, 0) # 2017 as a reference

DAYS = []
for lead in fcst_leads_ini:
    date_temp = date_base + timedelta(hours=lead)
    DAYS.append(date_temp.day-1)
DAYS = np.array(DAYS[2:56])

# converting post-processed GEFS to flags and then probabilities
mon_inds = []
for d, date in enumerate(date_list):
    mon_inds.append(date.month-1)        
mon_inds = np.array(mon_inds)

prob = fcst_to_prob(AnEn_stn, BCH_90th, mon_inds)
binary = fcst_to_flag(BCH_obs, BCH_90th, mon_inds)

o_bar = np.empty((N_lead_day,))
use = np.empty((N_lead_day, N_bins))

prob_true = np.empty((N_lead_day, N_bins, N_boost))
prob_pred = np.empty((N_lead_day, N_bins, N_boost))
brier = np.empty((N_lead_day, N_boost,))

for r in range(3):
    for d in range(N_lead_day):
        
        day_ind = DAYS == d
        
        obs = binary[:, day_ind, :][..., FLAGs[r]].flatten()
        fcst = prob[:, day_ind, :][..., FLAGs[r]].flatten()

        flag_nonan = np.logical_not(np.isnan(obs))
        obs = obs[flag_nonan]
        fcst = fcst[flag_nonan]
        L = np.sum(flag_nonan)

        o_bar_ = np.mean(obs)

        o_bar[d] = o_bar_
        
        for n in range(N_boost):
            
            ind_bagging = np.random.choice(L, size=L, replace=True)
            obs_ = obs[ind_bagging]
            fcst_ = fcst[ind_bagging]
            
            prob_true_, prob_pred_ = reliability_diagram(obs_, fcst_, hist_bins)
            brier_ = brier_score_loss(obs_, fcst_)
            
            prob_true[d, :, n] = prob_true_
            prob_pred[d, :, n] = prob_pred_
            brier[d, n] = brier_
            
        hist_bins_ = np.mean(prob_pred[d, ...], axis=1)
        use_, _ = np.histogram(fcst, bins=np.array(list(hist_bins_)+[1.0]))
        use[d, :] = use_
            
    tuple_save = (brier, prob_true, prob_pred, use, o_bar)
    label_save = ['brier', 'pos_frac', 'pred_value', 'use', 'o_bar']
    du.save_hdf5(tuple_save, label_save, save_dir, '{}_Calib_loc{}.hdf'.format(prefix_out, r))

