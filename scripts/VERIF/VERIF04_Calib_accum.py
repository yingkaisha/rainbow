import sys
import argparse
from glob import glob
from datetime import datetime, timedelta

import h5py
import numpy as np
#import pandas as pd
from sklearn.metrics import brier_score_loss

sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/utils/')
sys.path.insert(0, '/glade/u/home/ksha/PUBLISH/fcstpp/')

#from fcstpp import metrics
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

# Defining params
parser = argparse.ArgumentParser()
parser.add_argument('lead', help='lead')
args = vars(parser.parse_args())

lead_ = int(args['lead'])

# three watershed groups
with h5py.File(save_dir+'BCH_wshed_groups.hdf', 'r') as h5io:
    flag_sw = h5io['flag_sw'][...]
    flag_si = h5io['flag_si'][...]
    flag_n = h5io['flag_n'][...]
FLAGs = (flag_sw, flag_si, flag_n)

with h5py.File(save_dir+'BCH_ERA5_3H_verif.hdf', 'r') as h5io:
    indx = h5io['indx'][...]
    indy = h5io['indy'][...]
    
with h5py.File(save_dir+'BCH_MODEL_cumsum.hdf', 'r') as h5io:
    BASE_accum = h5io['BASE_cumsum'][:, lead_, ...]
    BCNN_accum = h5io['BCNN_cumsum'][:, lead_, ...]
    SL_accum = h5io['SL_cumsum'][:, lead_, ...]
    SCNN_accum = h5io['SCNN_cumsum'][:, lead_, ...]
    BCH_obs_accum = h5io['BCH_obs_cumsum'][:, lead_, ...]
    
# importing domain info
with h5py.File(save_dir+'BC_domain_info.hdf', 'r') as h5io:
    land_mask_bc = h5io['land_mask_bc'][...]

# importing girdded ERA5 quantiles
with h5py.File(ERA_dir+'PT_accum_q_stn.hdf', 'r') as h5io:
    CDF_obs = h5io['era_accum_q'][...]
    q_bins = h5io['q_bins'][...]

BCH_90th = CDF_obs[:, 89, lead_, :]

N_days = 365 + 365 + 365
date_base = datetime(2017, 1, 1)
date_list = [date_base + timedelta(days=x) for x in np.arange(N_days, dtype=np.float)]


# param and allocation
# params
N_bins = 15
N_boost = 100
hist_bins = np.linspace(0, 1, N_bins)

prob_true_base = np.empty((N_bins, N_boost))
prob_true_bcnn = np.empty((N_bins, N_boost))
prob_true_sl = np.empty((N_bins, N_boost))
prob_true_scnn = np.empty((N_bins, N_boost))

prob_pred_base = np.empty((N_bins, N_boost))
prob_pred_bcnn = np.empty((N_bins, N_boost))
prob_pred_sl = np.empty((N_bins, N_boost))
prob_pred_scnn = np.empty((N_bins, N_boost))

brier_base = np.empty((N_boost,))
brier_bcnn = np.empty((N_boost,))
brier_sl = np.empty((N_boost,))
brier_scnn = np.empty((N_boost,))

# loop over three watersheds
for r in range(3):
    flag_ = FLAGs[r]
    BCH_accum_ = np.copy(BCH_obs_accum[..., flag_])
    BASE_accum_ = np.copy(BASE_accum[..., flag_])
    SL_accum_ = np.copy(SL_accum[..., flag_])
    BCNN_accum_ = np.copy(BCNN_accum[..., flag_])
    SCNN_accum_ = np.copy(SCNN_accum[..., flag_])

    BCH_thres_mon = BCH_90th[:, flag_]
    BCH_thres = np.empty((N_days, np.sum(flag_)))
    
    for i, date in enumerate(date_list):
        mon_ind = date.month - 1
        BCH_thres[i, :] = BCH_thres_mon[mon_ind, :]
        
    BCH_binary = (BCH_accum_ > BCH_thres).flatten()
    BASE_prob = (np.mean(1.0*(BASE_accum_ > BCH_thres[:, None, :]), axis=1)).flatten()
    BCNN_prob = (np.mean(1.0*(BCNN_accum_ > BCH_thres[:, None, :]), axis=1)).flatten()
    SL_prob = (np.mean(1.0*(SL_accum_ > BCH_thres[:, None, :]), axis=1)).flatten()
    SCNN_prob = (np.mean(1.0*(SCNN_accum_ > BCH_thres[:, None, :]), axis=1)).flatten()
    
    flag_pick = np.logical_not(np.isnan(BCH_binary))
    
    BCH_binary = BCH_binary[flag_pick]
    BASE_prob = BASE_prob[flag_pick]
    BCNN_prob = BCNN_prob[flag_pick]
    SL_prob = SL_prob[flag_pick]
    SCNN_prob = SCNN_prob[flag_pick]
        
    L = np.sum(flag_pick)

    for n in range(N_boost):
        ind_bagging = np.random.choice(L, size=L, replace=True)

        BCH_binary_ = BCH_binary[ind_bagging]
        BASE_prob_ = BASE_prob[ind_bagging]
        BCNN_prob_ = BCNN_prob[ind_bagging]
        SL_prob_ = SL_prob[ind_bagging]
        SCNN_prob_ = SCNN_prob[ind_bagging]

        prob_true_base_, prob_pred_base_ = reliability_diagram(BCH_binary_, BASE_prob_, hist_bins)
        prob_true_bcnn_, prob_pred_bcnn_ = reliability_diagram(BCH_binary_, BCNN_prob_, hist_bins)
        prob_true_sl_, prob_pred_sl_ = reliability_diagram(BCH_binary_, SL_prob_, hist_bins)
        prob_true_scnn_, prob_pred_scnn_ = reliability_diagram(BCH_binary_, SCNN_prob_, hist_bins)

        brier_base_ = brier_score_loss(BCH_binary_, BASE_prob_)
        brier_bcnn_ = brier_score_loss(BCH_binary_, BCNN_prob_)
        brier_sl_ = brier_score_loss(BCH_binary_, SL_prob_)
        brier_scnn_ = brier_score_loss(BCH_binary_, SCNN_prob_)

        prob_true_base[:, n] = prob_true_base_
        prob_true_bcnn[:, n] = prob_true_bcnn_
        prob_true_sl[:, n] = prob_true_sl_
        prob_true_scnn[:, n] = prob_true_scnn_

        prob_pred_base[:, n] = prob_pred_base_
        prob_pred_bcnn[:, n] = prob_pred_bcnn_
        prob_pred_sl[:, n] = prob_pred_sl_
        prob_pred_scnn[:, n] = prob_pred_scnn_

        brier_base[n] = brier_base_
        brier_bcnn[n] = brier_bcnn_
        brier_sl[n] = brier_sl_
        brier_scnn[n] = brier_scnn_
        
    o_bar = np.mean(BCH_binary)
        
    hist_bins_base = np.mean(prob_pred_base, axis=1)
    hist_bins_bcnn = np.mean(prob_pred_bcnn, axis=1)
    hist_bins_sl = np.mean(prob_pred_sl, axis=1)
    hist_bins_scnn = np.mean(prob_pred_scnn, axis=1)

    use_base, _ = np.histogram(BASE_prob, bins=np.array(list(hist_bins_base)+[1.0]))
    use_bcnn, _ = np.histogram(BCNN_prob, bins=np.array(list(hist_bins_bcnn)+[1.0]))
    use_sl, _ = np.histogram(SL_prob, bins=np.array(list(hist_bins_sl)+[1.0]))
    use_scnn, _ = np.histogram(SCNN_prob, bins=np.array(list(hist_bins_scnn)+[1.0]))
    
    tuple_save = (prob_true_base, prob_true_bcnn, prob_true_sl, prob_true_scnn, 
                  prob_pred_base, prob_pred_bcnn, prob_pred_sl, prob_pred_scnn,
                  brier_base, brier_bcnn, brier_sl, brier_scnn,
                  use_base, use_bcnn, use_sl, use_scnn, o_bar)

    label_save = ['prob_true_base', 'prob_true_bcnn', 'prob_true_sl', 'prob_true_scnn', 
                  'prob_pred_base', 'prob_pred_bcnn', 'prob_pred_sl', 'prob_pred_scnn',
                  'brier_base', 'brier_bcnn', 'brier_sl', 'brier_scnn',
                  'use_base', 'use_bcnn', 'use_sl', 'use_scnn', 'o_bar']
    
    du.save_hdf5(tuple_save, label_save, save_dir, 'Accum_Calib_lead{}_loc{}.hdf'.format(lead_, r))
    