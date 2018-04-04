import matplotlib
matplotlib.use('TkAgg')

import logging

try:
    from config_autolabeller import getconfig
except ImportError:
    print ('Failed to import Orbslam2Service')
else:
    print('Orbslam2Service imported')
