import os
import sys
import matplotlib
from pprint import pprint
matplotlib.use('TkAgg')


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

# this_dir = os.path.dirname(__file__)
# print('this_dir: %s' % this_dir)

print('sys.path:')
pprint(sys.path)
