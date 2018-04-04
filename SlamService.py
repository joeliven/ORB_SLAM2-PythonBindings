#!/usr/bin/python
import _init_paths
from config_slam import getconfig
from utils.file_utils import mkdir_p, get_filenames, s3_upload_file, download_aws_file, download_aws_files
from utils.logging_utils import _log, _send_slack_status_log_print
from utils.name_utils import parse_sqs_msg, get_sqs_msg, get_unique_path, \
    get_logname, get_timestamp, get_prefix, get_dets_path, \
    video0bucket_KEY, video0key_KEY, imagesbucket_KEY, prefix_KEY, timestamp0_KEY, timestamp1_KEY, scanID_KEY, droneID_KEY, modelID_KEY, TMPDIR_KEY, msgstr_KEY, CAMMTX_KEY, FPS_KEY
from utils.yaml_utils import load_camera_matrix, loadSettings, saveSettings
from algo_wrappers.Orbslam2 import Orbslam2

import cv2
import boto3
from collections import deque
import json
from multiprocessing.pool import Pool
import argparse
import pickle
import uuid
import shutil
import time
import os
from copy import deepcopy
from pprint import pprint
import logging
lg = logging.getLogger()
lg.setLevel(logging.INFO)
import pdb

cfg = getconfig()
slack_channel = cfg.logging.info.slack.channel
slack_hook_url = cfg.logging.info.slack.url
os.environ['slackChannel'] = slack_channel
os.environ['SlackHookUrl'] = slack_hook_url


def download_sqs_msg_and_images(bucket_name, logging_prefix):
    """Download images from an AWS S3 bucket
    @note NOTA BENE: errors here will not surface to main thread!
    """
    try:
        sqs_tmp_dir = os.path.join('/tmp', '{}'.format(uuid.uuid4()))
        _log('attempting to download saved sqs msg to %s' % sqs_tmp_dir)
        sqs_key, sqs_download_path = download_aws_file(bucket_name, sqs_tmp_dir, logging_prefix, suffix='sqs.txt', type='saved sqs msg', send_status_update=True, status_header='*SLAM/PREPROCESSING:*')
        if sqs_key is not None:
            sqs_msg_str = []
            with open(sqs_download_path, 'r')  as f:
                for line in f:
                    sqs_msg_str.append(line.strip())
            _log('len(sqs_msg_str): %d' % len(sqs_msg_str))
            sqs_msg_str = ''.join(sqs_msg_str).strip()
            _log('sqs_msg_str: %s' % sqs_msg_str)

            # parse the downloaded sqs msg into a dict
            scan_info = parse_sqs_msg(sqs_msg_str)

            # NB: this is a key difference from the RecognitionEngine.py's version of this function.
            # it is critical here that timestamp is taken from the scan_info dict so that the tags
            # in the autolabelling correspond to the timestamp of the scan video!!!
            timestamp0 = scan_info[timestamp0_KEY]
            timestamp1 = deepcopy(timestamp0)

            scanID = scan_info[scanID_KEY]
            droneID = scan_info[droneID_KEY]
            modelID = scan_info[modelID_KEY]
            imagesbucket = scan_info[imagesbucket_KEY]
            prefix = scan_info[prefix_KEY]
            # clean up by removing the tmp dir that the saved sqs msg file was downloaded to since no longer needed:
            shutil.rmtree(sqs_tmp_dir)
        else:
            # no sqs msgs found in this logging bucket with this prefix
            _log('WARNING: no saved sqs msg found in this bucket (%s) with this prefix (%s) and this suffix (sqs.txt). SLAM cannot run, so ignoring this job.' % (bucket_name, logging_prefix))
            return None
        # get tmp_dir downloading the dets file:
        tmp_dir = get_unique_path(timestamp1, scanID, tmp=True, auto=True)
        scan_info[TMPDIR_KEY] = tmp_dir
        scan_info[timestamp0_KEY] = timestamp0
        scan_info[timestamp1_KEY] = timestamp1

        if imagesbucket:
            image_keys, image_download_paths = download_aws_files(imagesbucket, tmp_dir, prefix, suffix=None, type='image', status_header='*SLAM/PREPROCESSING:*', send_status_update=True)
        else:
            txt = '*SLAM/PREPROCESSING:* Unable to find images associated with sqs msg. Image downloading skipped. SLAM job will be aborted...'
            _send_slack_status_log_print(text=txt)

        return scan_info
    except Exception as e:
        print(str(e))
        raise


def delete_bucket_images(bucket_name, prefix):
    """Delete images from an AWs S3 bucket
    
    @param imgBucket A boto3.Bucket of images
    @param imgPrefix The S3 "directory" to delete
    """
    # Create a new resource for each worker to avoid interference
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    toDelete = [{'Key': o.key} for o in bucket.objects.filter(Prefix=prefix)]
    bucket.delete_objects(Delete={'Objects': toDelete})


class SlamService(object):
    """Standalone executable for SlamService service.

    The SlamService service awaits newly uploaded sqs messages
    from the Lambda preprocessing service, which has performed
    image preprocessing (frame extraction from video,
    image undistortion, image cropping - if desired - etc.).
    This SlamService service then downloads the images and
    associated with the scan signalled by the sqs message
    and runs the specified slam algo on the scan.
    The camera trajectory (as a sequence of poses, one per frame)
    are then uploaded to the appropriate AWS S3 bucket,
    and the Pensa Cognitive Engine is then signalled via an SQS message.
    """
    
    def __init__(self):
        self.cfg = getconfig()
        self.log_bucket_name = self.cfg.logging.info.s3.bucket
        self.pool = Pool(processes=int(self.cfg.slam.bucket_transfer_pool))
        self.scan_dirs = deque()
        self.slamAlgo = Orbslam2(vocabPath=self.cfg.orbslam2.vocab_path)

    def aws_init(self):
        """Configure AWs and connect to resources
        """
        sqs = boto3.resource('sqs')
        self.queue = sqs.get_queue_by_name(QueueName=self.cfg.slam.q_name)
        self.queue_results = sqs.get_queue_by_name(QueueName=self.cfg.slam.q_name_results)

    def check_queue(self):
        """
        :return:
        """
        msg = self.queue.receive_messages(
            WaitTimeSeconds=self.cfg.slam.q_poll_interval,
            MaxNumberOfMessages=1)
        if msg:
            txt = '*SLAM/PREPROCESSING:* Received sqs message.'
            _send_slack_status_log_print(text=txt)
            bucket_path_ = msg[0].body
            bucket_path = bucket_path_.split(',')
            if len(bucket_path) == 2:
                bucket_name, prefix = bucket_path[0],bucket_path[1]
            else:
                err_msg = "sqs slam msg deformed: must be in form of `<bucket_name>,<path/to/file>` "\
                      "but msg body was: %s" % bucket_path_
                print(err_msg)
                _send_slack_status_log_print(text=err_msg)
                msg[0].delete()
                return
            print('prefix: %s' % prefix)
            self.pool.apply_async(download_sqs_msg_and_images,
                                  args=(bucket_name, prefix),
                                  callback=self.new_scan_callback)
            msg[0].delete()

    def new_scan_callback(self, result):
        """Callback for completed pool download tasks
        @param scan_info dict
        """
        if result is None:
            return
        video0bucket = result[video0bucket_KEY]
        video0key = result[video0key_KEY]
        imagesbucket = result[imagesbucket_KEY]
        prefix = result[prefix_KEY]
        timestamp0 = result[timestamp0_KEY]
        timestamp1 = result[timestamp1_KEY]
        scanID = result[scanID_KEY]
        droneID = result[droneID_KEY]
        modelID = result[modelID_KEY]
        tmp_dir = result[TMPDIR_KEY]
        msg_str = result[msgstr_KEY]

        camMtxBucket = self.cfg.slam.camMtx.bucket
        camMtxKey = self.cfg.slam.camMtx.keys.get(droneID, 'sdf000003_drone197_v001.xml')
        _, camMtxPath = download_aws_file(camMtxBucket, download_dir=tmp_dir, prefix=camMtxKey, suffix=None, type=None, status_header='*SLAM/PREPROCESSING:*', send_status_update=False)
        result[CAMMTX_KEY] = camMtxPath
        # Logging...
        client = boto3.client('s3')
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(self.log_bucket_name)
        # save the sqs msg with the scan_info_string to a text file and upload to s3 + slack post
        # so that the full sqs message can easily be resent for reprocessing with or w/o recognition:
        self.log_sqs_msg(client, bucket, timestamp0, timestamp1, scanID, msg_str, tmp_dir)
        found_or_not = 'Retrieved'
        image_str = 'images'
        txt =   '*SLAM/PREPROCESSING:* %s %s for scanID: %s\n' \
                '\tscan video bucket:\t%s\n' \
                '\tscan video key:\t\t%s\n' \
                '\timages bucket:\t\t%s\n' \
                '\timages prefix:\t\t%s\n' \
                '\ttimestamp0:\t\t\t%s\n' \
                '\ttimestamp1:\t\t\t%s\n' \
                '\tdroneID:\t\t\t%s\n' \
                '\tmodelID:\t\t\t%s\n' \
                % (found_or_not, image_str, str(scanID), video0bucket, video0key, imagesbucket, prefix, str(timestamp0), str(timestamp1), str(droneID), str(modelID))
        _send_slack_status_log_print(txt)

        self.scan_dirs.appendleft(result)

    def signal_completion(self, scan_info):
        """
        Notify the rest of the system that the slam service has finished processing this scan,
        by sending an sqs message to the pensa-localization-slam sqs Q that the
        pensa Cognitive Engine polls on to retrieve the resulting trajectory output
        from the slam service.
        :param scan_info:
        :return:
        """
        res = self.queue_results.send_message(MessageBody=msg)
        _log(res.get('MessageId'))

        pass
        # RIGHT HERE: TODO: signal the extraction service to grab crops from autolabelled images and upload
        # to specified classifier datasets.

        # video0bucket = scan_info[video0bucket_KEY]
        # video0key = scan_info[video0key_KEY]
        # imagesbucket = scan_info[imagesbucket_KEY]
        # prefix = scan_info[prefix_KEY]
        # timestamp = scan_info[timestamp_KEY]
        # scanID = scan_info[scanID_KEY]
        # droneID = scan_info[droneID_KEY]
        # modelID = scan_info[modelID_KEY]
        #
        # sqs_msg_str = get_sqs_msg(video0bucket,
        #                           video0key,
        #                           imagesbucket,
        #                           prefix,
        #                           timestamp,
        #                           scanID,
        #                           droneID,
        #                           modelID)
        #
        # # Send a message signaling the stand-alone autolabelling service that a new scan is ready to be autolabelled:
        # _log('msg: %s' % sqs_msg_str)
        # txt = '*RE/AUTOLABELLING:* Sending sqs msg to signal Autolabelling service...'
        # _send_slack_status_log_print(txt)
        # res = self.queue_autolabel.send_message(MessageBody=sqs_msg_str)
        # _log(res.get('MessageId'))

    def pop_scan_dir(self):
        """
        Pop the oldest scan_info dict of the scan dir and return it for processing.
        :return: scan_info_d: dict of info about the scan coming from the received sqs msg.
        """
        scan_info_d = self.scan_dirs.pop()
        return scan_info_d

    def log_sqs_msg(self, client, bucket, timestamp0, timestamp1, scanID, sqs_msg, imgdir):
        """
        Upload the sqs msg in string format to a text file in the autolabellers logging bucket.
        Also uploads a temporary link to that file to the slack status-autolabeller channel.
        :param client: aws client.
        :param bucket: aws bucket object.
        :param timestamp0: str: time of original scan.
        :param timestamp1: str: time of original scan (same in this instance but is not always the case when used in RE).
        :param scanID: str: possibly uuid4 style scan id.
        :param sqs_msg: str: string formatted version of the sqs msg to upload.
        :param imgdir: tmp directorty in which artifacts of this autolabelling job are being stored.
        :return:
        """
        logname = get_logname(timestamp1, scanID)
        logname_sqs = get_logname(timestamp0, scanID)
        logname_sqs += '_sqs.txt'

        sqs_path = os.path.join(imgdir, logname_sqs)
        with open(sqs_path, 'w') as f:
            f.write(sqs_msg)
        sqs_key = os.path.join(logname, logname_sqs)
        sqs_s3_url = s3_upload_file(client, bucket, key=sqs_key, path=sqs_path, content_type='text')
        sqs_msg = '*SLAM/LOGGING*: <%s|sqs message>' % (sqs_s3_url)
        _send_slack_status_log_print(text=sqs_msg)

    def log_trajectory(self, client, bucket, timestamp0, timestamp1, scanID, trajectoryPath):
        """
        Upload the dets file used for executing this autolabelling job (pickle file format) to the autolabellers logging bucket.
        Also uploads a temporary link to that file to the slack status-autolabeller channel.
        :param client: aws client.
        :param bucket: aws bucket object.
        :param timestamp0: str: time of original scan.
        :param timestamp1: str: time of original scan (same in this instance but is not always the case when used in RE).
        :param scanID: str: possibly uuid4 style scan id.
        :param trajectoryPath: path to the saved trajectory file.
        :return:
        """
        if trajectoryPath is None or not os.path.isfile(trajectoryPath):
            _log('trajectoryPath %s does not exist or is not a file.' % str(trajectoryPath))
        logname = get_logname(timestamp1, scanID)
        logname_trajectory = get_logname(timestamp0, scanID)
        logname_trajectory += '_trajectory.txt'

        trajectory_key = os.path.join(logname, logname_trajectory)
        trajectory_s3_url = s3_upload_file(client, bucket, key=trajectory_key, path=trajectoryPath, content_type='text/plain')
        trajectory_msg = '*SLAM/LOGGING*: <%s| trajectory file>' % (trajectory_s3_url)
        _send_slack_status_log_print(text=trajectory_msg)

    def submit_logs(self, timestamp0, timestamp1, scanID, imgdir, trajectoryPath=None, log_which={'all'}):
        """
        Submit whichever logs are indicated by `log_which`. See individual log submission functions for details on each of those.
        :param timestamp0: str: time of original scan.
        :param timestamp1: str: time of original scan (same in this instance but is not always the case when used in RE).
        :param scanID: str: possibly uuid4 style scan id.
        :param imgdir: tmp directorty in which artifacts of this autolabelling job are being stored.
        :param trajPath: path to the saved trajectory file.
        :param log_which: set of str: indicates which items to log.
        :return:
        """
        # Logging...
        client = boto3.client('s3')
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(self.log_bucket_name)

        if 'all' in log_which:
            log_which = {'trajectory'}
        # upload the trajectory of poses (as a text file) to s3 + slack post
        if self.cfg.logging.get('trajectory') and 'trajectory' in log_which:
            self.log_trajectory(client, bucket, timestamp0, timestamp1, scanID, trajectoryPath)

    def updateSettingsFile(self, camMtx, fps, saveDir, templatePath=None):
        """
        This function makes a copy of the template settings file, makes the necessary mods based on the
        supplied camera matrix and fps, saves the settings file as a new file and returns the path of
        the saved file.
        :param camMtx: 3x3 camera intrisics matrix
        :param fps: frame rate in frames per second
        :param templatePath: path to the template settings file
        :param saveDir: dir in which to save the modified settings file
        :return: settingsPath: the path to the updated settings file
        """
        if templatePath is None:
            templatePath = self.cfg.orbslam2.settings_template_path
        settingsD = loadSettings(templatePath)
        fxi = self.cfg.slam.camMtx.layout.FXi
        fxj = self.cfg.slam.camMtx.layout.FXj
        fyi = self.cfg.slam.camMtx.layout.FYi
        fyj = self.cfg.slam.camMtx.layout.FYj
        cxi = self.cfg.slam.camMtx.layout.CXi
        cxj = self.cfg.slam.camMtx.layout.CXj
        cyi = self.cfg.slam.camMtx.layout.CYi
        cyj = self.cfg.slam.camMtx.layout.CYj
        fx = camMtx[fxi,fxj]
        fy = camMtx[fyi,fyj]
        cx = camMtx[cxi,cxj]
        cy = camMtx[cyi,cyj]
        settingsD['Camera.fx'] = fx
        settingsD['Camera.fy'] = fy
        settingsD['Camera.cx'] = cx
        settingsD['Camera.cy'] = cy
        settingsD['Camera.fps'] = fps

        savePath = os.path.join(saveDir, 'orbslam2_settings.yaml')
        saveSettings(settingsD, savePath)
        return savePath

    def run(self):
        """Main process loop

        Checks the SQS message queue intermittently for newly uploaded
        images. Upon seeing a new message, the images are downloaded
        locally, and then we run them through slam algo.
        """
        self.aws_init()
        
        while True:
            if self.cfg.slam.active:
                self.check_queue()

            print('scan_dirs: ' + str(len(self.scan_dirs)))
            if len(self.scan_dirs) > 0:
                try:
                    scan_info = self.pop_scan_dir()
                    print('scan_info')
                    pprint(scan_info)

                    video0bucket = scan_info[video0bucket_KEY]
                    video0key = scan_info[video0key_KEY]
                    imagesbucket = scan_info[imagesbucket_KEY]
                    prefix = scan_info[prefix_KEY]
                    timestamp0 = scan_info[timestamp0_KEY]
                    timestamp1 = scan_info[timestamp1_KEY]
                    scanID = scan_info[scanID_KEY]
                    droneID = scan_info[droneID_KEY]
                    modelID = scan_info[modelID_KEY]
                    imagesdir = scan_info[TMPDIR_KEY]
                    fps = scan_info.get(FPS_KEY, 10)
                    camMtxPath = scan_info[CAMMTX_KEY]

                    # Load the camMtx so we can update the settings file with the Fx, Fy, Cx, Cy, and fps values
                    # and then save the updated file and pass along the path to this updated settings file
                    # to the orbslam2 wrapper object:
                    camMtx = load_camera_matrix(camMtxPath)
                    settingsPath = self.updateSettingsFile(camMtx, fps, saveDir=imagesdir)

                    image_paths = sorted(get_filenames(imagesdir, ext=tuple(self.cfg.slam.image_exts)))
                    nb_image_files = len(image_paths)

                    if nb_image_files > 0:
                        trajectory, trajectorySavePath = self.slamAlgo.runSlam(image_paths, settingsPath, fps, useViewer=False)
                        self.submit_logs(timestamp0, timestamp1, scanID, imagesdir, trajectoryPath=trajectorySavePath, log_which={'transforms'})


                    # clean up:
                    if os.path.exists(imagesdir):
                        shutil.rmtree(imagesdir)

                except Exception as e:
                    print(str(e))
                    raise


# def loadSettings(settingsPath):
#     fs = cv2.FileStorage(settingsPath, cv2.FILE_STORAGE_READ)
#     fn = fs.getNode("camera_matrix")


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Stand alone executable for the Orbslam2 Service.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    slamService = SlamService()
    slamService.run()


if __name__ == "__main__":
    main()