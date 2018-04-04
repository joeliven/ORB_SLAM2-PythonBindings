"""Simple utility functions for standardizing name conventions."""
try:
    from utils.logging_utils import _log
except ImportError as e:
    print(str(e))
    from logging_utils import _log
import os
from datetime import datetime
from dateutil import parser
import time
SEP = '___'
TMP_DIR = '/tmp'
DETS_DIR = 'dets_data'
AUTO_DIR = 'slam'
SUFFIXES = {'.h264','.mp4','.avi','.mov',   # video extensions
            '.jpg', '.jped', '.png',        # image extensions
            '.pickle', '.pkl',              # pickle extensions
            '.csv'                          # csv result file extensions
            }
video0bucket_KEY = 'video0bucket'
video0key_KEY = 'video0key'
imagesbucket_KEY = 'imagesbucket'
prefix_KEY = 'prefix'
timestamp0_KEY = 'timestamp'
timestamp1_KEY = 'timestamp1'
scanID_KEY = 'scanID'
droneID_KEY = 'droneID'
modelID_KEY = 'modelID'
TMPDIR_KEY = 'tmpDir'
msgstr_KEY = 'msgStr'
CAMMTX_KEY = 'camMtx'
FPS_KEY= 'fps'


def get_image_ext(image_path):
    image_ext = '.' + image_path.split('.')[-1]
    return image_ext


def get_image_dets_dir(image_dir):
    image_dets_dir = image_dir + '_dets'
    return image_dets_dir


def get_logname(timestamp, scanID, suffix=None, sep='_'):
    logname = get_unique_path(timestamp, scanID)
    if suffix is not None:
        if not (isinstance(suffix, str) and isinstance(sep, str)):
            raise ValueError('Both suffix and sep must be of type `str` but are type `%s` and `%s`' % (str(type(suffix)), str(type(sep))))
        logname = '%s%s%s' % (logname, sep, suffix)
    return logname


def get_dets_path(imgdir, timestamp, scanID):
    logname = get_logname(timestamp, scanID, suffix='dets')
    dets_path = os.path.join(imgdir, logname + '.pickle')
    return dets_path


def get_unique_path(timestamp, scanID, tmp=False, auto=False):
    if scanID:
        timestamp_scanUUID = '%s%s%s' % (str(timestamp), SEP, scanID)
    else:
        timestamp_scanUUID = timestamp

    if auto:
        timestamp_scanUUID = os.path.join(AUTO_DIR, timestamp_scanUUID)
    if tmp:
        return os.path.join(TMP_DIR, timestamp_scanUUID)
    else:
        return timestamp_scanUUID


def get_prefix(timestamp, scanUUID, drone_id, model_id, tmp=False):
    timestamp_scanUUID = '%s%s%s' % (str(timestamp), SEP, scanUUID)
    if tmp:
        return os.path.join(TMP_DIR, timestamp_scanUUID, drone_id, model_id)
    else:
        return os.path.join(timestamp_scanUUID, drone_id, model_id)


def parse_prefix(prefix):
    if prefix.startswith(TMP_DIR):
        # remove '/tmp' prefix from path:
        prefix = prefix.split(TMP_DIR)[1]
    head, tail = os.path.split(prefix)
    for suf in SUFFIXES:
        # remove the filename portion of the path if it is a file not a dir:
        if suf.lower() in tail.lower():
            prefix = head
            break
    parts = prefix.split('/')
    assert len(parts) == 3, '%s\tlen(parts): %d' % ('/'.join(parts), len(parts))
    timestamp_scanUUID = parts[0]
    droneID = parts[1]
    modelID = parts[2]
    timestamp_scanUUID = timestamp_scanUUID.split(SEP)
    assert len(timestamp_scanUUID) == 2, '%s\tlen(timestamp_scanUUID): %d' % (SEP.join(timestamp_scanUUID), len(timestamp_scanUUID))
    timestamp, scanUUID = timestamp_scanUUID
    return timestamp, scanUUID, droneID, modelID


def parse_unique_path(path):
    if path.startswith(TMP_DIR):
        # remove '/tmp' prefix from path:
        path = path.split(TMP_DIR)[1]
    head, tail = os.path.split(path)
    for suf in SUFFIXES:
        # remove the filename portion of the path if it is a file not a dir:
        if suf.lower() in tail.lower():
            path = head
            break
    timestamp_scanUUID = path.split(SEP)
    assert len(timestamp_scanUUID) == 2, '%s\tlen(timestamp_scanUUID): %d' % (SEP.join(timestamp_scanUUID), len(timestamp_scanUUID))
    timestamp, scanUUID = timestamp_scanUUID
    return timestamp, scanUUID


def get_sqs_autolabel_msg(log_bucket, timestamp, scanID):
    key = get_unique_path(timestamp, scanID)
    msg = ','.join([log_bucket, key])
    return msg


def get_sqs_msg(video0bucket, video0key, imagesbucket, prefix, timestamp0, timestamp1, scanID, droneID, modelID, fps=10):
    msg_parts = [
        '%s=%s' % (video0bucket_KEY,video0bucket),
        '%s=%s' % (video0key_KEY,video0key),
        '%s=%s' % (imagesbucket_KEY,imagesbucket),
        '%s=%s' % (prefix_KEY,prefix),
        '%s=%s' % (timestamp0_KEY, timestamp0),
        '%s=%s' % (timestamp1_KEY, timestamp1),
        '%s=%s' % (scanID_KEY,scanID),
        '%s=%s' % (droneID_KEY,droneID),
        '%s=%s' % (modelID_KEY,modelID),
        '%s=%s' % (FPS_KEY,fps)
    ]
    msg = ','.join(msg_parts)
    return msg


def parse_sqs_msg(msg):
    msg_parts = msg.split(',')
    parsed = {}
    for part in msg_parts:
        key,val = part.split('=')
        parsed[key] = val
    parsed[msgstr_KEY] = msg
    return parsed
    # video0_bucket, video0_key, images_bucket, prefix, timestamp, scanID, droneID, modelID = vals


def get_timestamp(tm=None):
    if tm is None:
        return datetime.now().strftime('%Y%m%d_%H%M%S')
    dt_obj = parse_timestamp(tm)
    timestamp = dt_obj.strftime('%Y%m%d_%H%M%S')
    return timestamp


def parse_timestamp(tm):
    if isinstance(tm, datetime):
        return tm # no-op since we are trying to parse whatever input tm is into a datetime object
    try:
        dt_obj = parser.parse(tm)
    except (ValueError,TypeError) as e:
        _log(str(e))
        try:
            if isinstance(tm, str):
                if '_' in tm:
                    tm = ' '.join(tm.split('_'))
                dt_obj = parser.parse(tm)
            elif isinstance(tm, (int,float)):
                dt_obj = parser.parse(time.strftime('%Y%m%d %H%M%S', time.localtime(tm)))
            else:
                raise TypeError('Not able to parse tm with type (%s).' % str(type(tm)))
        except (ValueError,TypeError) as e:
            _log(str(e))
            raise
    if not isinstance(dt_obj, datetime):
        raise TypeError('tm must be convertible to a datetime object but is not (type of dt_obj: %s).') % str(type(dt_obj))
    return dt_obj