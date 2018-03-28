import errno
import os
import zipfile as zipf
from io import BytesIO
import pickle
# import boto3
import time
from operator import itemgetter


def get_filenames(raw_dir, ext=None, names_only=False):
    """
    get all filenames (as full filepaths) in raw_dir that end in ext
    :param raw_dir:     str:                        directory to get files listed in
    :param ext:         str or tuple of strings:    extension(s) to list files for (files ending in anything else are ignored)
    :return:            list of strings:            list of filenames in raw_dir ending in ext
    """
    if not os.path.exists(raw_dir):
        return []
    if ext is None:
        if names_only:
            raw_files = [f for f in os.listdir(raw_dir) if os.path.isfile(os.path.join(raw_dir, f))]
        else:
            raw_files = [os.path.join(raw_dir,f) for f in os.listdir(raw_dir) if os.path.isfile(os.path.join(raw_dir, f))]
    else:
        assert isinstance(ext, (str,tuple)), 'ext (type=%s) must be of type string or a tuple of strings' % str(type(ext))
        if names_only:
            raw_files = [f for f in os.listdir(raw_dir) if (os.path.isfile(os.path.join(raw_dir, f)) and f.lower().endswith(ext))]
        else:
            raw_files = [os.path.join(raw_dir,f) for f in os.listdir(raw_dir) if (os.path.isfile(os.path.join(raw_dir, f)) and f.lower().endswith(ext))]
    return raw_files


def get_dirnames(raw_dir, names_only=False):
    """
    get all dir_names (as full filepahts) in raw_dir
    :param raw_dir: str:                directory to get files listed in
    :return:        list of strings:    list of filenames in raw_dir ending in ext
    """
    if not os.path.exists(raw_dir):
        return []
    if names_only:
        raw_files = [f for f in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, f))]
    else:
        raw_files = [os.path.join(raw_dir,f) for f in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, f))]
    return raw_files


def get_dirs_and_files(raw_dir, ext=None, names_only=False):
    """
    get all dir or file names (as full filepahts) in raw_dir (for files, optionally can require that they end in ext)
    :param raw_dir: str:                        directory to get files and dirs listed in
    :param ext:     str or tuple of strings:    extension(s) to list files for (files ending in anything else are ignored)
    :return:        list of strings:            list of filenames (files or dirs) in raw_dir (if files, then must ebd in ext if ext is not None)
    """
    if ext is None:
        if names_only:
            raw_files = [f for f in os.listdir(raw_dir) if (os.path.isfile(os.path.join(raw_dir, f)) or os.path.isdir(os.path.join(raw_dir, f)))]
        else:
            raw_files = [os.path.join(raw_dir,f) for f in os.listdir(raw_dir) if (os.path.isfile(os.path.join(raw_dir, f)) or os.path.isdir(os.path.join(raw_dir, f)))]
    else:
        assert isinstance(ext, (str,tuple)), 'ext (type=%s) must be of type string or a tuple of strings' % str(type(ext))
        if names_only:
            raw_files = [f for f in os.listdir(raw_dir) if
                (os.path.isfile(os.path.join(raw_dir, f)) or
                 (os.path.isdir(os.path.join(raw_dir, f)) and f.lower().endswith(ext))
                 )]
        else:
            raw_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if
                (os.path.isfile(os.path.join(raw_dir, f)) or
                 (os.path.isdir(os.path.join(raw_dir, f)) and f.lower().endswith(ext))
                 )]

    return raw_files


def mkdir_p(path):
    try:
        os.makedirs(path)
        print('making path: %s' % path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def zip_files(filepaths):
    print('zip_files')
    mf = BytesIO()
    with zipf.ZipFile(mf, 'w') as zf:
        for filepath in filepaths:
            zf.write(filepath, compress_type=zipf.ZIP_DEFLATED)
    mf.seek(0)
    return mf


def zip_files_on_disk(filepaths, savepath):
    print('zip_files_on_disk')
    zf = zipf.ZipFile(savepath, 'w')
    try:
        for filepath in filepaths:
            zf.write(filepath, compress_type=zipf.ZIP_DEFLATED)
    finally:
        zf.close()
    return zf


def crop_pickle(loadpath, savepath=None, ltrim=0, rtrim=0):
    with open(loadpath, 'rb') as handle:
        pickle_data = pickle.load(handle)
    assert len(pickle_data) > (ltrim + rtrim + 1), 'cannot trim more than 1 less than length of data: ' \
                                                   'len(pickle_data): %d    rtrim: %d    ltrim: %d' % (len(pickle_data), ltrim, rtrim)
    pickle_data = pickle_data[ltrim: -rtrim]
    if savepath is None:
        if '.pickle' in loadpath:
            savepath = loadpath.replace('.pickle', '_cropped.pickle')
        elif '.pkl' in loadpath:
            savepath = loadpath.replace('.pkl', '_cropped.pkl')
        else:
            savepath = loadpath + '.cropped'

    with open(savepath, 'wb') as handle:
        pickle.dump(pickle_data, handle)

    return savepath


def s3_upload_file(client, bucket, key, path, content_type, expires=36000, presigned=True):
    bucket.upload_file(path, key, ExtraArgs={'ContentType': content_type})
    if presigned:
        # return presigned url:
        return client.generate_presigned_url('get_object', Params={'Bucket': bucket.name, 'Key': key}, ExpiresIn=expires)


def s3_presigned_url(client, bucket, key, expires=36000):
    # return presigned url:
    return client.generate_presigned_url('get_object', Params={'Bucket': bucket.name, 'Key': key}, ExpiresIn=expires)


def s3_upload_file_obj(client, bucket, key, fileobj, content_type, expires=36000, presigned=True):
    bucket.upload_fileobj(Fileobj=fileobj, Key=key, ExtraArgs={'ContentType': content_type})
    if presigned:
        # return presigned url:
        return client.generate_presigned_url('get_object', Params={'Bucket': bucket.name, 'Key': key}, ExpiresIn=expires)


# def download_aws_file(bucket_name, download_dir, prefix, suffix=None, type=None, status_header='*RE/PREPROCESSING:*', send_status_update=False):
#     if type is None:
#         if suffix is not None:
#             type = suffix
#         else:
#             type = 'unspecified'
#
#     txt = '%s Finding %s file(s) from bucket _%s_ with prefix _%s_ and suffix _%s_ and downloading to _%s_' % (status_header, type, bucket_name, prefix, suffix, download_dir)
#     _send_slack_status_log_print(text=txt, send_status_update=send_status_update)
#
#     start = time.time()
#
#     # Create a new resource for each worker to avoid interference
#     s3 = boto3.resource('s3')
#     bucket = s3.Bucket(bucket_name)
#
#     download_paths = []
#     keys = []
#     for o in bucket.objects.filter(Prefix=prefix):
#         filename = o.key
#         print('filename: %s' % filename)
#         if suffix and not filename.endswith(suffix):
#             continue
#         filename = os.path.split(filename)[1]
#         download_path = os.path.join(download_dir, filename)
#         keys.append(o.key)
#         download_paths.append(download_path)
#
#     nb_files = len(download_paths)
#     if len(download_paths) > 0:
#         mkdir_p(download_dir)
#         if nb_files > 1:
#             _log('Warning: %d %s files found with prefix %s and suffix %s' % (nb_files, type, prefix, suffix))
#         keys_paths = zip(keys, download_paths)
#         sorted_keys_paths = sorted(keys_paths, key=itemgetter(0))
#         keys, download_paths = zip(*sorted_keys_paths)
#         key = keys[-1]
#         download_path = download_paths[-1]
#         txt = '%s Downloading %d of %d possible %s file(s) to %s....' % (status_header, 1, nb_files, type, download_path)
#         _send_slack_status_log_print(text=txt, send_status_update=send_status_update)
#         bucket.download_file(key, download_path)
#         dur = time.time() - start
#         txt = '%s *_%s_* found and downloaded in %f seconds.' % (status_header, key, dur)
#         _send_slack_status_log_print(text=txt, send_status_update=send_status_update)
#     else:
#         key, download_path = None, None
#         txt = '%s Failed to find any matching %s files in:\n\tbucket:\t%s\n\tprefix:\t%s\n\tsuffix:\t%s' % (status_header, type, bucket_name, prefix, suffix)
#         _send_slack_status_log_print(text=txt, send_status_update=send_status_update)
#     return key, download_path
#
#
# def download_aws_files(bucket_name, download_dir, prefix, suffix=None, type=None, status_header='*RE/PREPROCESSING:*', send_status_update=False):
#     if type is None:
#         if suffix is not None:
#             type = suffix
#         else:
#             type = 'unspecified'
#
#     txt = '%s Downloading %s files to %s' % (status_header, type, download_dir)
#     _send_slack_status_log_print(text=txt, send_status_update=send_status_update)
#
#     start = time.time()
#
#     # Create a new resource for each worker to avoid interference
#     s3 = boto3.resource('s3')
#     bucket = s3.Bucket(bucket_name)
#
#     download_paths = []
#     keys = []
#     mkdir_p(download_dir)
#     for o in bucket.objects.filter(Prefix=prefix):
#         filename = o.key
#         if suffix and not filename.endswith(suffix):
#             continue
#         filename = os.path.split(filename)[1]
#         download_path = os.path.join(download_dir, filename)
#         bucket.download_file(o.key, download_path)
#         keys.append(o.key)
#         download_paths.append(download_path)
#
#     nb_files = len(download_paths)
#     dur = time.time() - start
#     avg = dur / float(nb_files) if nb_files > 0 else 0.
#     txt = '%s Done downloading %d %s files in %f seconds (%f / file).' % (status_header, nb_files, type, dur, avg)
#     _send_slack_status_log_print(text=txt)
#     return keys, download_paths


