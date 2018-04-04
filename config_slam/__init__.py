from yaml import load
from copy import deepcopy
import sys
import os
from pprint import pprint

sys.path.insert(0, os.path.abspath('..'))
from easydict import EasyDict as edict

global cfg
__configpath__ = os.path.realpath(os.path.join(os.getcwd(),
                                               os.path.dirname(__file__),
                                               'config.orbslam2.yaml'))
print('\n\n##################################################################################################')
print('##################################################################################################')
print('################################# ORBSLAM2 SERVICE ENVIRONEMENT    ##################################   ')
print('##################################################################################################')
print('##################################################################################################')

with open(__configpath__, 'r') as f:
    cfg = edict(load(f))

print('Loaded config settings from: %s' % __configpath__)
pprint(cfg)
print('##################################################################################################')
print('##################################################################################################')
print('##################################################################################################')
print('##################################################################################################\n\n\n')


def getconfig():
    """Returns global config
    """
    return edict(cfg)
