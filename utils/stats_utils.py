from utils.file_utils import mkdir_p

import matplotlib.pyplot as plt
import numpy as np
from collections import deque, defaultdict
import os


def softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x."""
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    e_x = np.exp(x - np.max(x, axis=axis))
    return e_x / e_x.sum()


def min_max(x, axis=0, invert=False):
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    if x.shape[axis] == 1:
        return np.ones_like(x) * 0.5
    min = np.min(x, axis=axis)
    max = np.max(x, axis=axis)

    delta = max - min
    normed = (x - min) / delta
    if invert:
        # 1. ---> 0.
        # 0. ---> 1.
        normed = 1.0 - normed
    return normed


def crop_distribution(data, thresh=0.):
    N = len(data)
    cur_start = 0
    cur_end = 0
    cur_sum = 0.
    max_start = 0
    max_end = 0
    max_sum = 0.
    for t,val in enumerate(data):
        if val <= thresh:                # not in the keep area, might be start or end though
            if cur_sum == 0:                # not start or end, so increment cur_start, cur_end
                cur_start = t
                cur_end = t
            elif cur_sum <= max_sum:        # end of a run, but not the biggest run, so reset
                cur_start = t
                cur_end = t
                cur_sum = 0.
            else:                           # end of a run, and it IS THE BIGGEST, so change max info to reflect this.
                max_start = cur_start
                max_end = t
                max_sum = cur_sum
                cur_sum = 0.
        else:                           # in the middle of a run, so update accordingly
            cur_end = t
            cur_sum += val

    # check for runs that end with cur_sum > 0
    if cur_sum > 0:
        if cur_sum <= max_sum:          # the final run was not the biggest, so reset
            cur_start = N-1
            cur_end = N-1
            cur_sum = 0.
        else:                           # the final run WAS THE BIGGEST, so reset, so change max info to reflect this.
            max_start = cur_start
            max_end = N-1
            max_sum = cur_sum
            cur_sum = 0.

    # increment end by 1 to account for exclusive numpy slicing:
    max_end += 1
    print('max_start: %d \tmax_end: %d \t max_sum: %d' % (max_start, max_end, max_sum))

    # sanity checks:
    assert max_sum >= cur_sum, 'max_sum (%f) not >= cur_sum (%f)' % (max_sum, cur_sum)
    assert max_end >= max_start, 'max_end (%d) not >= max_start (%d)' % (max_end, max_start)
    assert cur_end >= cur_start, 'cur_end (%d) not >= cur_start (%d)' % (cur_end, cur_start)

    # crop the distribution to zero out everything except the max run found above:
    data[0:max_start] *= 0.
    data[max_start: max_end] *= 1.
    data[max_end :] *= 0.

    return data


def weighted_moving_average(x, window=10, weights=None, pad=0.):
    if not isinstance(x, np.ndarray):
        x = np.asarray(x, dtype=np.float32)
    if x.dtype not in {np.float32, np.float64}:
        x = np.asarray(x, np.float32)
    if weights is not None:
        weights = np.asarray(weights, dtype=x.dtype)
        if weights.shape[0] == window:
            dynamic_weights = False
        elif weights.shape[0] == x.shape[0]:
            dynamic_weights = True
        else:
            msg = 'weights must be a 1D array-like with shape equal to window (%d) or ' \
                  'with same shape as x (%s) but has shape %s' % (window, str(x.shape), str(weights.shape))
            raise ValueError(msg)
    else:
        dynamic_weights = False

    N = x.shape[0]
    if window % 2 == 0:
        lwin = int(window / 2) - 1  # left window is one smaller
        rwin = int(window / 2)  # right window
    else:
        lwin = int(window / 2)  # left window
        rwin = int(window / 2)  # right window
    assert lwin + rwin + 1 == window, 'lwin (%d) + rwin (%d) + 1 != window (%d)' % (lwin, rwin, window)

    x_avg = np.zeros_like(x)
    for center in range(0, x.shape[0]):
        start = center - lwin
        end = center + rwin
        # grab the valid portion of the slice, ensuring no OutOfIndex errors:
        slice = x[max(0, start): min(N-1, end)]
        if dynamic_weights:
            w_slice = weights[max(0, start): min(N-1, end)]

        if start < 0:
            # handle first lwin number of entries in array where
            # we cannot center the window without pre padding
            pre = np.ones(shape=(abs(start),), dtype=x.dtype)
            slice = np.concatenate([pre * float(pad), slice], axis=0)
            if dynamic_weights:
                w_slice = np.concatenate([pre, w_slice], axis=0)

        if end >= N:
            # handle last rwin number of entries in array where
            # we cannot center the window without post padding
            post = np.ones(shape=(end - (N-1),), dtype=x.dtype)
            slice = np.concatenate([slice, post * float(pad)], axis=0)
            if dynamic_weights:
                w_slice = np.concatenate([w_slice, post], axis=0)
        if dynamic_weights:
            assert w_slice.shape == slice.shape, 'w_slice.shape (%s)   slice.shape (%s)' % (str(w_slice.shape), str(slice.shape))
            avg = np.average(slice, axis=0, weights=w_slice)
        else:
            avg = np.average(slice, axis=0, weights=weights)
        x_avg[center] = avg
    return x_avg


def weight_dets(dets_data, window=20, weights=None, remove_low_conf_dets=True, savedir=None):
    """
    Compute an estimated unormalized discrete time density function for each class, and weight the
    confidence of the detections based on this time density estimation.
    NB: the estimated discrete density function is not normalized in the same sense that a typical
    probability distribution is normalized - meaning the sum of it's mass is not equal to 1.
    However it is "quasi-normalized" in the sense that every discrete point of mass in the density
    estimation is scaled to the [0. to 1.] range inclusive.
    :param dets_data: list of dicts, one for each image in the scan. Each dict is structured like:
        dets_data[i] = {'confs':confs, 'bboxes': bboxes, 'image_path':image_path}
        where confs is:
            a dict mapping prodID --> to an ndarray of shape (nb_dets,) that represents the raw conf score for each detection
        where bboxes is:
            a dict mapping prodID --> to an ndarray of shape (nb_dets, 4) that represents the bbox for each detection

    :return:
    """
    # since subclassifiers were not always forced to give zero dets arrays
    # (ie confs=np.zeros(shape=(0,)) and zero bboxes array (ie bboxes=np.zeros(shape=(0,4))
    # for those of their subclasses that did not appear in an image -- NB: they are now! --
    # much of the historical dets data requires that we explicity grab all prodIDs to use
    # as keys, otherwise the data structures become uneven and we get IndexErrors.
    allProdIDs_l = [set(img_dets['confs'].keys()) for img_dets in dets_data]
    allProdIDs = set()
    for someProdIDs in allProdIDs_l:
        allProdIDs = allProdIDs | someProdIDs
    allProdIDs = sorted(allProdIDs)

    prodIDcounts = defaultdict(list)
    prodIDavgConfs = defaultdict(list)
    for image_dets in dets_data:
        bboxes = image_dets['bboxes']
        confs = image_dets['confs']

        for prodID in allProdIDs:
            prodConfs = confs.get(prodID, np.zeros(shape=(0,), dtype=np.float32))
            prodConfs = prodConfs ** 4
            avgProdConfs = np.average(prodConfs, axis=0) if prodConfs.shape[0] > 0 else 1.
            nbDets = prodConfs.shape[0]
            prodIDcounts[prodID].append(nbDets)
            prodIDavgConfs[prodID].append(avgProdConfs)

    prodIDcountsAvgd = {}
    for prodID in allProdIDs:
        detCts = np.asarray(prodIDcounts[prodID], dtype=np.float32)
        avgConfs = prodIDavgConfs[prodID]
        detCtsAvgd = weighted_moving_average(detCts, window=window, weights=avgConfs)
        # normalize weighted moving averages so that all fall in range [0.  -  1.] inclusive:
        maxCtWeighted = np.amax(detCtsAvgd)
        if maxCtWeighted > 0:
            detCtsAvgd /= maxCtWeighted

        detCtsAvgd = crop_distribution(data=detCtsAvgd, thresh=0.)
        # detCtsAvgd = softmax(detCtsAvgd)
        prodIDcountsAvgd[prodID] = detCtsAvgd

    allProdTimeDensityWeights = prodIDcountsAvgd
    if savedir is not None:
        mkdir_p(savedir)
        for prodID in sorted(allProdTimeDensityWeights.keys()):
            data = allProdTimeDensityWeights[prodID]
            plt.plot(data)
            plt.ylabel('dets per frame (moving weighted avg)')
            plt.xlabel('frame number')
            plt.title('prodID: %d' % prodID)
            # plt.show()
            savepath = os.path.join(savedir, '%d.png' % prodID)
            plt.savefig(savepath)
            plt.close()

    weighted_dets_data = []
    for image_num,image_dets in enumerate(dets_data):
        bboxes = image_dets['bboxes']
        confs = image_dets['confs']

        weighted_image_dets = {}
        weighted_image_dets['bboxes'] = bboxes
        weighted_confs = {}
        for prodID in sorted(confs.keys()):  # TODO: remove "sorted()" after debugging; strictly used for reproducibiilty.
            prodTimeDensityWeights = allProdTimeDensityWeights[prodID]
            prodTimeDensityWeight = prodTimeDensityWeights[image_num]
            assert isinstance(prodTimeDensityWeight, (float, np.float32, np.float64)), 'type(prodTimeDensityWeight) is %s' % str(type(prodTimeDensityWeight))
            prodConfs = confs[prodID]
            weightedProdConfs = prodConfs * prodTimeDensityWeight
            if np.sum(weightedProdConfs) == 0 and remove_low_conf_dets:
                weightedProdConfs = np.zeros(shape=(0,), dtype=np.float32)
                weighted_image_dets['bboxes'][prodID] = np.zeros(shape=(0,4), dtype=np.float32)
            weighted_confs[prodID] = weightedProdConfs
        weighted_image_dets['confs'] = weighted_confs
        weighted_dets_data.append(weighted_image_dets)

    return weighted_dets_data



