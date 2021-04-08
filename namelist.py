import numpy as np

# ========== Parameters ========== #
N_fcst = 54
FCSTs = np.arange(9.0, 24*7+3, 3)
# period = 3
# FCSTs = np.arange(9.0, 24*9+period, period)
# FCSTs = FCSTs[:N_fcst]

# ========== General path ========== #
# Data and backup
DATA_dir = '/glade/scratch/ksha/DATA/'
BACKUP_dir = '/glade/scratch/ksha/BACKUP/'
drive_dir = '/glade/scratch/ksha/DRIVE/'

# ========== Data ========== #

# NAEFS
NAEFS_dir = DATA_dir+'NAEFS/'

# GEFS reforecast
REFCST_dir = DATA_dir+'REFCST/'
REFCST_source_dir = drive_dir+'/REFCST/refcstv12/'
GFS_APCP_single_dir = REFCST_source_dir+'apcp/'
GFS_PWAT_single_dir = REFCST_source_dir+'pwat/'

# ERA5 reanalysis
ERA_dir = BACKUP_dir + 'ERA5/ERA5_PCT/'

# PRISM
PRISM_dir = BACKUP_dir + 'PRISM/'

# ======= Neural network ======= #

# Path of model check point
temp_dir = '/glade/work/ksha/data/Keras/'

# Path of batch files
BATCH_dir = DATA_dir+'REFCST_PCT/'
BATCH_dir2 = DATA_dir+'REFCST_PCT2/'

# ========== AnEn-CNN ========== #
# SL search domain indices (; GEFS and ERA5)
## [lat0, lat1, lon0, lon1]
domain_inds = [120, 280, 120, 340]

# BC domain indices
## [lat0, lat1, lon0, lon1]
bc_inds = [73, 121, 36, 148]

# Evaluation results
save_dir = temp_dir + 'BIAS_publish/'

# ========== Graphics ========== #

# figure storage
fig_dir = '/glade/u/home/ksha/figures/'

# Matplotlib figure export settings
fig_keys = {'dpi':250, 
            'orientation':'portrait', 
            'papertype':'a4',
            'bbox_inches':'tight', 
            'pad_inches':0.1, 
            'transparent':False}

# colors
rgb_array = np.array([[0.85      , 0.85      , 0.85      , 1.        ],
                      [0.66666667, 1.        , 1.        , 1.        ],
                      [0.33333333, 0.62745098, 1.        , 1.        ],
                      [0.11372549, 0.        , 1.        , 1.        ],
                      [0.37647059, 0.81176471, 0.56862745, 1.        ],
                      [0.10196078, 0.59607843, 0.31372549, 1.        ],
                      [0.56862745, 0.81176471, 0.37647059, 1.        ],
                      [0.85098039, 0.9372549 , 0.54509804, 1.        ],
                      [1.        , 1.        , 0.4       , 1.        ],
                      [1.        , 0.8       , 0.4       , 1.        ],
                      [1.        , 0.53333333, 0.29803922, 1.        ],
                      [1.        , 0.09803922, 0.09803922, 1.        ],
                      [0.8       , 0.23921569, 0.23921569, 1.        ],
                      [0.64705882, 0.19215686, 0.19215686, 1.        ],
                      [0.55      , 0.        , 0.        , 1.        ]])
    
blue   = rgb_array[3, :]  # blue
cyan   = rgb_array[2, :]  # cyan
lgreen = rgb_array[4, :]  # light green
green  = rgb_array[5, :]  # dark green
yellow = rgb_array[8, :]  # yellow
orange = rgb_array[-6, :] # orange
red    = rgb_array[-3, :] # red
