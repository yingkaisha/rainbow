'''
Functions for processing BC Hydro observation files.
*Raw files are in PST with daylight savings.
'''

import numpy as np
import pandas as pd
from os.path import basename
from datetime import datetime, timedelta

def flag_out(dataframe, col, flag_col, flag_val):
    '''
    Preserve values that have qc_code=`flag_val`; flag out the rest.
        'PREC_INST_RAW_QC' should have flag_val='200'
        'PREC_INST_QC_QC'  should have flag_val='50'
    '''
    temp_data = dataframe[col].values
    temp_flag = dataframe[flag_col].values
    temp_data[temp_flag!=flag_val] = np.nan
    dataframe[col] = temp_data
    return dataframe


def BCH_txt_preprocess(filename_input, filename_output, cols_num, qc_code, verbose=True):
    '''
    Converting BC Hydro (old) txt observation files into pandas HDF5.
    
    Input
    ----------
        filename_input: a list of raw txt file names
        filename_output: the output filename, a string, e.g., 'NRT_2016_2018.hdf'
        cols_num: list of namea of numerical columns, e.g., 'PREC_INST_QC'
        verbose: True for print outs
        qc_code: code of good quality, the same length as cols_num
        
    Output
    ----------
        keys: a list of stations been processed
        keys_stored: a list of stations been stored (have non-NaN values)
        
    '''
    
    cols_need = []
    for col in cols_num:
        cols_need.append(col)
        cols_need.append(col+'_QC') # add qc code into extracted cols
    cols_need = ['datetime',] + cols_need
    keys = []; keys_stored = []
    
    with pd.HDFStore(filename_output, 'w') as hdf_io:
        # loop over txt files
        for name in filename_input:
            # Assuming good at the begining
            Bad = False 

            # txt to pandas
            data = pd.read_csv(name, sep='\t', skiprows=[0])

            # use filename (first three letters) as hdf variable keys
            key = basename(name)[0:3]
            keys.append(key);
            if verbose:
                print(key)

            # collection all detected columns
            cols = list(data.columns)

            # rename the datetime column
            cols[1] = 'datetime' 
            data.columns = cols

            # check missing in the needed columns
            for col_need in cols_need:
                if np.logical_not(col_need in cols):
                    if verbose:
                        print('\t"{}" missing'.format(col_need))
                    Bad = True;
            if Bad:
                if verbose:
                    print('\tInsufficient data')
                continue;

            # subset to needed columns
            data_subset = data[cols_need].copy()
            
            # collecting numerical values from needed columns
            L = len(data_subset)
            for col_num in cols_num:
                temp_data = np.empty(L) # allocation
                temp_string = data_subset[col_num]
                for i, string in enumerate(temp_string):
                    # column could be a 'float number' or '+' for NaN
                    try:
                        temp_data[i] = np.float(string)
                    except:
                        # "try: failed = got "+" symbol
                        temp_data[i] = np.nan

                # replace raw strings by the converted values
                data_subset[col_num] = temp_data
            
            # flag out values based on the qc code
            if qc_code is not None:
                for i, col in enumerate(cols_num):
                    data_subset = flag_out(data_subset, col, col+'_QC', flag_val=qc_code[i])
            
            # drop rows that contain NaN values
            data_subset = data_subset.dropna(how='any')

            # if found "0.0" in datetime col, mark as NaN and drop the row 
            for i, date_vals in enumerate(data_subset['datetime']):
                if date_vals == "0.0":
                    if verbose:
                        print("\tFound bad datetime values, drop row {}".format(i))
                    data_subset = data_subset.drop(data_subset.index[i])

            # converting datetime string to pandas datetime after cleaning
            data_subset['datetime'] = pd.to_datetime(data_subset['datetime'])

            # check the number of remained columns
            L = len(data_subset)
            if L < 2:
                if verbose:
                    print('\tInsufficient data after value cleaning, L={}'.format(L))
                continue;

            # observational times as ending times
            # calculating the time diff for resmapling 
            freq = np.empty(L)
            for i in range(L-1):
                freq[i+1] = (data_subset['datetime'].iloc[i+1] - data_subset['datetime'].iloc[i]).total_seconds()
            freq[0] = freq[1]
            data_subset['FREQ'] = freq

            # dropna changes index, here reset the pandas index
            data_out = data_subset.reset_index().drop(['index'], axis=1)

            # rename all pre-processed columns
            data_out.columns = cols_need + ['FREQ',]

            # save into the hdf
            hdf_io[key] = data_out
            keys_stored.append(key)

    return keys, keys_stored

def BCH_xls_preprocess(filename_input, filename_output, cols_num, verbose=True):
    '''
    Converting BC Hydro xls observation files into pandas HDF5.
    
    Input
    ----------
        filename_input: a list of xls file names
        filename_output: the output filename, a string, e.g., 'NRT_2019.hdf'
        cols_num: name of the single numerical column, e.g., 'PREC_INST_QC'
        verbose: True for print outs
        
    Output
    ----------
        keys: a list of stations been processed
        
    '''
    
    keys = []
    with pd.HDFStore(filename_output, 'w') as hdf_io:
        for name in filename_input:
            pd_temp = pd.read_excel(name)
            
            # get col names, expecting ['Date', 'stn code']
            col_names = list(pd_temp.columns)
            stn_name = col_names[1]
            keys.append(stn_name)
            
            if verbose:
                print('{}'.format(stn_name))
            
            # allocate a new DataFrame for output
            pd_out = pd.DataFrame()
            
            # assigning values
            pd_out['datetime'] = pd.to_datetime(pd_temp[col_names[0]])
            pd_out[cols_num] = pd_temp[col_names[1]]
            
            # drop NaN
            pd_out = pd_out.dropna()
            
            # save
            hdf_io[stn_name] = pd_out
            
    return keys

def dt_to_sec(dt):
    '''
    python datetime to datenum relative to 1970-01-01.
    '''
    L = len(dt)
    base = datetime(1970, 1, 1)
    out = [0]*L
    for i, t in enumerate(dt):
        out[i] = int((t-base).total_seconds())
    return out

def BCH_PREC_resample(bucket_height, sec_obs, date_start, date_end, period=60*60):
    '''
    Coverting BC Hydro NRT bucket heights into precipitation rate [mm/period].
    
    Note: this function works similarly to the BC Hydro resample scheme, 
          but filters out more values [produces more NaNs, see (1), (2), and (3)].
    
    Input
    ----------
        bucket_height: observed bucket height
        sec_obs: observed seconds relative to 1970-1-1
        date_start, date_end: start and end datetime to be resampled.
                              *date_end is not included, i.e., [start, end)
        period: seconds of resampling period, default is hourly.

    Output
    ----------
        Resampled precipitation rate [mm/period]
        A list of datetimes as reference
        *Missing/bad values are filled with nan
        
    '''
    
    # start and end datetime relative to 1970-01-01
    sec_ref_min = dt_to_sec([date_start,])[0]
    sec_ref_max = dt_to_sec([date_end,])[0]
    
    # datetime to be resampled
    sec_base = np.arange(sec_ref_min, sec_ref_max, period)
    
    # length of before and after resampling
    L_time = len(sec_base)
    L_obs = len(bucket_height)

    # (1) Bad instrument flag: 
    #     If bucket height is negative, mark it as bad
    bucket_height[bucket_height<0] = np.nan


    delta = 0
    neg_count = 0

    bucket_height_fix = [] #np.empty((L_obs,))
    bucket_height_fix.append(bucket_height[0])

    sec_obs_fix = []
    sec_obs_fix.append(sec_obs[0])

    for i in range(1, L_obs):

        # (2) No response flag:
        #     If two bucket heights have their time gap > resample_preiod, mark the half-gap as bad 
        if sec_obs[i] - sec_obs[i-1] > 2*period:
            # insert a np.nan
            dt = sec_obs[i] - sec_obs[i-1]
            bucket_height_fix.append(np.nan)
            sec_obs_fix.append(sec_obs[i] - 0.5*dt)

        # (3) Evaporation flag: 
        #     If negative bucket height diff appeared two times, mark the later as bad
        if bucket_height[i] < bucket_height[i-1]:
            neg_count += 1
            delta += bucket_height[i-1] - bucket_height[i]
        else:
            neg_count = 0

        if neg_count >= 2:
            bucket_height_fix.append(np.nan)
            sec_obs_fix.append(sec_obs[i])
        else:
            bucket_height_fix.append(bucket_height[i] + delta)
            sec_obs_fix.append(sec_obs[i])

    bucket_height_fix = np.array(bucket_height_fix)
    sec_obs_fix = np.array(sec_obs_fix)

    sec_full = np.arange(sec_ref_min, sec_ref_max, 1)
    bucket_height_interp = np.interp(sec_full, sec_obs_fix, bucket_height_fix)
    bucket_height_fold = bucket_height_interp.reshape(-1, period)

    precip = np.empty((L_time,))
    precip[0] = np.nan
    for i in range(1, L_time):
        precip[i] = bucket_height_fold[i, 0] - bucket_height_fold[i-1, 0]

    # ---------- Datetime ---------- #
    date_ref = [date_start + timedelta(seconds=x) for x in range(0, sec_ref_max-sec_ref_min, period)]
    
    return precip, date_ref
