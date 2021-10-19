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

EN = 45

# ========== BCH obs preprocessing ========== # 

# import station obsevations and grid point indices
with h5py.File(save_dir+'BCH_ERA5_3H_verif.hdf', 'r') as h5io:
    BCH_obs = h5io['BCH_obs'][...]
    indx = h5io['indx'][...]
    indy = h5io['indy'][...]
    
# subsetting BCH obs into a given year
N_days = 366 + 365*3
date_base = datetime(2016, 1, 1)
date_list = [date_base + timedelta(days=x) for x in np.arange(N_days, dtype=np.float)]

flag_pick = []
for date in date_list:
    if date.year in [2017, 2018, 2019]:
        flag_pick.append(True)
    else:
        flag_pick.append(False)

flag_pick = np.array(flag_pick)
    
BCH_obs = BCH_obs[flag_pick, ...]

# ========== ERA5 stn climatology preprocessing ========== #

# importing domain info
with h5py.File(save_dir+'BC_domain_info.hdf', 'r') as h5io:
    land_mask_bc = h5io['land_mask_bc'][...]

# importing girdded ERA5 quantiles
with h5py.File(ERA_dir+'PT_3hour_q.hdf', 'r') as h5io:
    CDF_era = h5io['era_3hq_bc'][...]
    q_bins = h5io['q_bins'][...]

CDF_obs = np.empty((12, 107,)+land_mask_bc.shape)
CDF_obs[..., ~land_mask_bc] = CDF_era
CDF_obs = CDF_obs[..., indx, indy]

# station and monthly (contains neighbouring months) wise 90th
BCH_90th = CDF_obs[:, (93+9), :] # <--- 90-th is selected as thres

# ========== PP preprocessing ========== #
# re calc datelist for 2017-2019
N_days = 365 + 365 + 365
date_base = datetime(2017, 1, 1)
date_list = [date_base + timedelta(days=x) for x in np.arange(N_days, dtype=np.float)]

N_stn = BCH_obs.shape[-1]

with h5py.File(save_dir+'NA_SL_info.hdf', 'r') as h5io:
    W_SL = h5io['W_SL'][bc_inds[0]:bc_inds[1], bc_inds[2]:bc_inds[3]][indx, indy]

AnEn = np.empty((N_days, N_fcst, EN, N_stn))

for y in [2017, 2018, 2019]:
    if y%4 == 0:
        n_days = 366
    else:
        n_days = 365
        
    flag_pick = []
    for date in date_list:
        if date.year == y:
            flag_pick.append(True)
        else:
            flag_pick.append(False)
    
    flag_pick = np.array(flag_pick)
    
    for lead in range(N_fcst):
        #print("lead = {}".format(lead))
        
        with h5py.File(REFCST_dir + "GEFS_QM_{}_lead{}.hdf".format(y, lead), 'r') as h5io:
            GEFS_ = h5io['gefs_qm'][:, :EN, ...][..., indx, indy]
            
        AnEn[flag_pick, lead, ...] = GEFS_

# ========== Calibration ========== #
# params
N_bins = 15
hist_bins = np.linspace(0, 1, N_bins)
N_boost = 100

# 3-hr lead times to days
fcst_leads_ini = np.arange(0, 72*3+3, 3, dtype=np.float)
date_base = datetime(2017, 1, 1, 0) # 2017 as a reference

DAYS = []
for lead in fcst_leads_ini:
    date_temp = date_base + timedelta(hours=lead)
    DAYS.append(date_temp.day-1)
DAYS = np.array(DAYS[2:56])

# number of forecasted days
N_lead_day = 7

# three watershed groups
with h5py.File(save_dir+'BCH_wshed_groups.hdf', 'r') as h5io:
    flag_sw = h5io['flag_sw'][...]
    flag_si = h5io['flag_si'][...]
    flag_n = h5io['flag_n'][...]
    
FLAGs = (flag_sw, flag_si, flag_n)

# converting post-processed GEFS to flags and then probabilities

mon_inds = []
for d, date in enumerate(date_list):
    mon_inds.append(date.month-1)        
mon_inds = np.array(mon_inds)

prob = fcst_to_prob(AnEn, BCH_90th, mon_inds)
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
    du.save_hdf5(tuple_save, label_save, save_dir, 'GEFS_99th_Calib_loc{}.hdf'.format(r))

