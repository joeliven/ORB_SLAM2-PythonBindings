import os
import sys
import matplotlib
matplotlib.use('TkAgg')
from pprint import pprint


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = os.path.dirname(__file__)

repo_root_path = os.path.join(this_dir, '..')
add_path(repo_root_path)

print('sys.path:')
pprint(sys.path)
