
import sys
import argparse
from glob import glob
from datetime import datetime, timedelta
from sklearn.metrics import brier_score_loss

# data tools
import h5py
import numba as nb
import numpy as np

# custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/Analog_BC/')

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

@nb.njit(fastmath=True)
def quantile_to_flag(data, quantile_grids, mon_inds):
    '''
    determine if gridded fcst/obs is larger than 
    the given grid-point-wise quantile values.  
    '''
    
    input_shape = data.shape
    output = np.empty(input_shape)
    output[...] = np.nan
    
    # no ensemble dimensions (time, grid)
    if len(input_shape) == 2:    
        for d, mon_ind in enumerate(mon_inds):
            output[d, :] = data[d, :] > quantile_grids[mon_ind, :]
            
    # (time, ensemble, grid)
    else:
        for d, mon_ind in enumerate(mon_inds):
            for en in range(input_shape[1]):
                output[d, en, :] = data[d, en, :] > quantile_grids[mon_ind, :]
    return output


parser = argparse.ArgumentParser()
parser.add_argument('out', help='out')
args = vars(parser.parse_args())

type_ind = int(args['out'])

if type_ind == 0:
    perfix_smooth = 'BASE_final'
    perfix_raw = 'BASE_final'
    key_smooth = 'AnEn_SG'
    key_raw = 'AnEn'
    
elif type_ind == 1:
    perfix_smooth = 'SL_final'
    perfix_raw = 'SL_final'
    key_smooth = 'AnEn_SG'
    key_raw = 'AnEn'
    
elif type_ind == 2:
    perfix_smooth = 'BASE_CNN'
    perfix_raw = 'BASE_final'
    key_smooth = 'cnn_pred'
    key_raw = 'AnEn'
    
elif type_ind == 3:
    perfix_smooth = 'SL_CNN'
    perfix_raw = 'SL_final'
    key_smooth = 'cnn_pred'
    key_raw = 'AnEn'
    
print("perfix_smooth = {}; perfix_raw = {}".format(perfix_smooth, perfix_raw))

with h5py.File(save_dir+'NA_SL_info.hdf', 'r') as h5io:
    W_SL = h5io['W_SL'][bc_inds[0]:bc_inds[1], bc_inds[2]:bc_inds[3]]

# Importing Geo-info
with h5py.File(save_dir+'BC_domain_info.hdf', 'r') as h5io:
    base_lon = h5io['base_lon'][...]
    base_lat = h5io['base_lat'][...]
    land_mask = h5io['land_mask_base'][...]
    land_mask_bc = h5io['land_mask_bc'][...]

# Defining params
N_fcst = 54
N_grids = np.sum(~land_mask_bc)

EN = 75
N_bins = 25
N_lead_day = 7

hist_bins = np.linspace(0, 1, N_bins)

# 2017-2020
year0 = 2017
year1 = 2020
N_days = 365*(year1-year0)

print('Preparing ERA5 obs')
era_tuple = ()
for year in range(year0, year1):
    with h5py.File(ERA_dir+'ERA5_GEFS-fcst_{}.hdf'.format(year), 'r') as h5io:
        era_ = h5io['era_fcst'][..., :N_fcst, bc_inds[0]:bc_inds[1], bc_inds[2]:bc_inds[3]]
    era_tuple += (era_[..., ~land_mask_bc],)
era_obs = np.concatenate(era_tuple, axis=0)

print('Preparing ensemble forecast')
PRED = np.empty((N_days, N_fcst, EN, N_grids))

for lead in range(N_fcst):
    print("\tlead = {}".format(lead))
    fcst_tuple = ()
    for year in range(year0, year1):
        with h5py.File(REFCST_dir + "{}_{}_lead{}.hdf".format(perfix_raw, year, lead), 'r') as h5io:
            RAW = h5io[key_raw][:, :EN, ...]
        with h5py.File(REFCST_dir + "{}_{}_lead{}.hdf".format(perfix_smooth, year, lead), 'r') as h5io:
            SMOOTH = h5io[key_smooth][:, :EN, ...]
        AnEn = W_SL*RAW + (1-W_SL)*SMOOTH
        fcst_tuple += (AnEn[..., ~land_mask_bc],)
    PRED[:, lead, ...] = np.concatenate(fcst_tuple, axis=0)
        
# comining 3-hr lead times to days
fcst_leads_ini = np.arange(0, 72*3+3, 3, dtype=np.float)
date_base = datetime(year0, 1, 1, 0)
DAYS = []
for lead in fcst_leads_ini:
    date_temp = date_base + timedelta(hours=lead)
    DAYS.append(date_temp.day-1)
DAYS = np.array(DAYS[2:56])

# map ini time to month of the year
base = datetime(year0, 1, 1)
date_list = [base + timedelta(days=x) for x in range(N_days)]
mon_inds = []
for date in date_list:
    mon_inds.append(date.month-1)
mon_inds = np.array(mon_inds)

# ERA quantiles (for 90th)
with h5py.File(ERA_dir+'PT_3hour_quantile.hdf', 'r') as h5io:
    CDF_era = h5io['CDF'][...]
    q_bins = h5io['q'][...]
    
# Evaluating fixed thresholds
for THRES in np.array([5, 30.0]):
    thres = THRES/8.0
    print("thres = {} mm/day, {} mm/3h".format(THRES, thres))
    prob_pred = np.sum(PRED>thres, axis=2)/75.0
    era_cate = era_obs > thres
    
    # allocation
    pos_frac_all = np.empty((N_lead_day, N_bins))
    pred_value_all = np.empty((N_lead_day, N_bins))
    use_all = np.empty((N_lead_day, N_bins))
    brier_all = np.empty((N_lead_day,))
    o_bar_all = np.empty((N_lead_day,))
    
    for i in range(N_lead_day):
        day_ind = DAYS == i
        prob_lead = prob_pred[:, day_ind, :].ravel()
        era_cate_lead = era_cate[:, day_ind, :].ravel()
        
        brier_all[i] = brier_score_loss(era_cate_lead, prob_lead)
        
        pos_frac, pred_value = reliability_diagram(era_cate_lead, prob_lead, hist_bins)
        use, _ = np.histogram(prob_lead, bins=np.array(list(pred_value)+[1.0]))
        o_bar = np.mean(era_cate_lead)

        pos_frac_all[i, :] = pos_frac
        pred_value_all[i, :] = pred_value
        use_all[i, :] = use
        o_bar_all[i,] = o_bar
        
    tuple_save = (brier_all, pos_frac_all, pred_value_all, use_all, o_bar_all)
    label_save = ['brier', 'pos_frac', 'pred_value', 'use', 'o_bar']
    du.save_hdf5(tuple_save, label_save, save_dir, '{}_Calib-{}mm_{}_{}.hdf'.format(perfix_smooth, int(THRES), year0, year1))

for p in np.array([0.9,]):
    print("thres = {} quantile values for all grid points.".format(p))
    quantile_grids = CDF_era[:, :-1, :]
    quantile_grids = quantile_grids[:, q_bins==p, :]
    
    PRED_binary = quantile_to_flag(PRED, quantile_grids, mon_inds)
    prob_pred = np.sum(PRED_binary, axis=2)/75.0
    era_cate = quantile_to_flag(era_obs, quantile_grids, mon_inds)
    
    # allocation
    pos_frac_all = np.empty((N_lead_day, N_bins))
    pred_value_all = np.empty((N_lead_day, N_bins))
    use_all = np.empty((N_lead_day, N_bins))
    brier_all = np.empty((N_lead_day,))
    o_bar_all = np.empty((N_lead_day,))
    
    for i in range(N_lead_day):
        day_ind = DAYS == i
        prob_lead = prob_pred[:, day_ind, :].ravel()
        era_cate_lead = era_cate[:, day_ind, :].ravel()

        brier_all[i] = brier_score_loss(era_cate_lead, prob_lead)

        pos_frac, pred_value = reliability_diagram(era_cate_lead, prob_lead, hist_bins)
        use, _ = np.histogram(prob_lead, bins=np.array(list(pred_value)+[1.0]))
        o_bar = np.mean(era_cate_lead)
        
        pos_frac_all[i, :] = pos_frac
        pred_value_all[i, :] = pred_value
        use_all[i, :] = use
        o_bar_all[i,] = o_bar
        
    tuple_save = (brier_all, pos_frac_all, pred_value_all, use_all, o_bar_all)
    label_save = ['brier', 'pos_frac', 'pred_value', 'use', 'o_bar']
    du.save_hdf5(tuple_save, label_save, save_dir, '{}_Calib-{}th_{}_{}.hdf'.format(perfix_smooth, int(p*100), year0, year1))
    
    
    