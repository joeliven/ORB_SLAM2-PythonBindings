#!/usr/bin/env python3
import _init_paths
import sys
import os
import orbslam2
import time
import cv2
from lib.utils.file_utils import get_filenames
from pprint import pprint


def main(vocab_path, settings_path, sequence_dir, fps=10):
    fps_inv = 1./float(fps)
    im_paths = sorted(get_filenames(sequence_dir, 'png'))
    timestamps = []
    for i in range(len(im_paths)):
        if i == 0:
            t = time.time()
        else:
            t = timestamps[i-1] + fps_inv
        timestamps.append(t)

    num_images = len(im_paths)

    slam = orbslam2.System(vocab_path, settings_path, orbslam2.Sensor.MONOCULAR)
    slam.set_use_viewer(False)
    slam.initialize()

    times_track = [0 for _ in range(num_images)]
    print('-----')
    print('Start processing sequence ...')
    print('Images in the sequence: {0}'.format(num_images))

    for idx in range(num_images):
        im_path = im_paths[idx]
        image = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
        tframe = timestamps[idx]

        if image is None:
            print("failed to load image at {0}".format(im_path))
            return 1

        t1 = time.time()
        pose_success = slam.process_image_mono(image, tframe)
        t2 = time.time()
        status = 'success' if pose_success else 'failure'
        print('idx %d: pose %s' % (idx, status))

        ttrack = t2 - t1
        times_track[idx] = ttrack

        t = 0
        if idx < num_images - 1:
            t = timestamps[idx + 1] - tframe
        elif idx > 0:
            t = tframe - timestamps[idx - 1]

        if ttrack < t:
            time.sleep(t - ttrack)

    traj = slam.get_trajectory_points()
    print('len(traj): %d' % len(traj))
    pprint(traj)
    save_trajectory(traj, 'trajectory.txt')

    slam.shutdown()

    times_track = sorted(times_track)
    total_time = sum(times_track)
    print('-----')
    print('median tracking time: {0}'.format(times_track[num_images // 2]))
    print('mean tracking time: {0}'.format(total_time / num_images))

    return 0


def save_trajectory(trajectory, filename):
    with open(filename, 'w') as traj_file:
        traj_file.writelines('{time} {r00} {r01} {r02} {t0} {r10} {r11} {r12} {t1} {r20} {r21} {r22} {t2}\n'.format(
            time=repr(stamp),
            r00=repr(r00),
            r01=repr(r01),
            r02=repr(r02),
            t0=repr(t0),
            r10=repr(r10),
            r11=repr(r11),
            r12=repr(r12),
            t1=repr(t1),
            r20=repr(r20),
            r21=repr(r21),
            r22=repr(r22),
            t2=repr(t2)
        ) for stamp, r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2 in trajectory)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('Usage: ./orbslam_mono_tum path_to_vocabulary path_to_settings path_to_sequence')
    main(sys.argv[1], sys.argv[2], sys.argv[3])
