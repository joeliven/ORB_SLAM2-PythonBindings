import cv2
import numpy as np
import pdb
from pprint import pprint
import os
import sys
import xml.etree.ElementTree as ET
import yaml
from easydict import EasyDict as edict
with open('config_slam/config.orbslam2.settings.yaml', 'r') as f:
    SETTINGS = edict(yaml.load(f)).orbslam2.settings


def openYaml(path, mode='r'):
    if mode == 'r':
        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    elif mode == 'w':
        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    else:
        raise ValueError
    return cv_file


def loadSettings(path):
    cv_file = openYaml(path, 'r')
    d = {}

    for k in SETTINGS.real:
        d[k] = cv_file.getNode(k).real()

    for k in SETTINGS.int:
        d[k] = int(cv_file.getNode(k).real())

    if SETTINGS.get('mat') is not None:
        for k in SETTINGS.mat:
            d[k] = cv_file.getNode(k).mat()
    cv_file.release()
    return d


def saveAsOldYaml(d, savePath):
    with open(savePath, 'w') as f:
        f.write('%YAML:1.0\n')
        for k,v in sorted(d.items()):
            f.write(k + ': ' + str(v) + '\n')


def saveSettings(d, savePath):
    saveAsOldYaml(d, savePath)


def load_camera_matrix_from_xml(camera_mtx_path):
    if not os.path.isfile(camera_mtx_path):
        raise ValueError('camera_mtx_path (%s) '
                         'must be a valid path to an openCV style xml file containg the data for a camera matrix. ' %
                         (str(camera_mtx_path)))
    camera_matrix = load_from_opencv_xml(filepath=camera_mtx_path, elementname='camera_matrix')
    assert camera_matrix.shape == (3, 3)
    print('camera_matrix loaded from xml: %s' % camera_mtx_path)
    pprint(camera_matrix)
    return camera_matrix


def load_from_opencv_xml(filepath, elementname, dtype='float32'):
    try:
        tree = ET.parse(filepath)
        rows = int(tree.find(elementname).find('rows').text)
        cols = int(tree.find(elementname).find('cols').text)
        return np.fromstring(tree.find(elementname).find('data').text, dtype, count=rows*cols, sep=' ').reshape((rows, cols))
    except Exception as e:
        print(str(e))
        raise


def load_camera_matrix(camMtxPath):
    return load_camera_matrix_from_xml(camMtxPath)


if __name__ == "__main__":
    settingsD = loadSettings('config_slam/pensa_orbslam2_settings_template.yaml')
    pprint(settingsD)
    # saveSettings(settingsD, 'config_slam/pensa_orbslam2_settings_template2.yaml')
