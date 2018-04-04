import os
import sys
from pprint import pprint
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import yaml

def parseXformsFile(path, nbFrames):
    xforms = []
    timestamps = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            parts = line.split()
            assert len(parts) == 13, 'len(parts): %d' % len(parts)
            parts = [float(p) for p in parts]
            timestamp = parts[0]
            r00 = parts[1]
            r01 = parts[2]
            r02 = parts[3]
            t0 = parts[4]
            r10 = parts[5]
            r11 = parts[6]
            r12 = parts[7]
            t1 = parts[8]
            r20 = parts[9]
            r21 = parts[10]
            r22 = parts[11]
            t2 = parts[12]
            xform = np.asarray([
                [r00, r01, r02, t0],
                [r10, r11, r12, t1],
                [r20, r21, r22, t2]
            ])
            timestamps.append(timestamp)
            xforms.append(xform)

    if len(xforms) > nbFrames:
        pass
#        raise ValueError('len(xforms) %d > nbFrames %d but shouldnt be.' % (len(xforms), nbFrames))
    if len(xforms) < nbFrames:
        nbMisses = nbFrames - len(xforms)
        misses =  [None for _ in range(nbMisses)]
        xforms = misses + xforms            # list concatenation
        timestamps = misses + timestamps    # list concatenation
    print('len(xforms): %d' % len(xforms))
    print('len(timestamps): %d' % len(timestamps))
 #   assert len(xforms) == nbFrames, 'len(xforms): %d != nbFrames %d but should' % (len(xforms), nbFrames)
    return xforms, timestamps


def convert_3x4_to_3x3(pose, camMtx, zPos=0.):
    """
    RefNote: https://stackoverflow.com/questions/13100573/how-can-i-transform-an-image-using-matrices-r-and-t-extrinsic-parameters-matric
    :param pose:
    :param camMtx:
    :param zPos:
    :return:
    """
    # converted condenses the 3x4 matrix which transforms a point in world space
    # to a 3x3 matrix which transforms a point in world space.  Instead of
    # multiplying pose by a 4x1 3d homogeneous vector, by specifying that the
    # incoming 3d vectors will ALWAYS have a z coordinate of zpos, one can instead
    # multiply converted by a homogeneous 2d vector and get the same output for x and y.
    """
    cv::Matx33f converted(pose(0,0),pose(0,1),pose(0,2)*zpos+pose(0,3),
                          pose(1,0),pose(1,1),pose(1,2)*zpos+pose(1,3),
                          pose(2,0),pose(2,1),pose(2,2)*zpos+pose(2,3));
    """
    zPos = float(zPos)
    converted = np.zeros((3,3), dtype=pose.dtype)
    converted[:,0:2] = pose[:,0:2].copy()
    converted[:,2] = pose[:,2]*zPos + pose[:,3]

    # This matrix will take a homogeneous 2d coordinate and "projects" it onto a
    # flat plane at zpos.  The x and y components of the incoming homogeneous 2d
    # coordinate will be correct, the z component is dropped.
    """
    cv::Matx33f projected(1,0,0,
                          0,1,0,
                          0,0,zpos);
    projected = projected*camera_mat.inv();
    """
    projected = np.eye(3,3, dtype=pose.dtype)
    projected[2,2] = zPos
    projected = np.matmul(projected, np.linalg.pinv(camMtx))
    # projected = projected * np.linalg.pinv(camMtx)

    # now we have the pieces.  A matrix which can take an incoming 2d point, and
    # convert it into a pseudo 3d point (x and y correspond to 3d, z is unused)
    # and a matrix which can take our pseudo 3d point and transform it correctly.
    # Now we just need to turn our transformed pseudo 3d point back into a 2d point
    # in our new image, to do that simply multiply by the camera matrix.

    # return camera_mat*converted*projected;
    return np.matmul(camMtx, np.matmul(converted, projected))
    # return camMtx * converted * projected


# def plot3Dposes(xforms):
#     refx = np.eye(4,4,dtype=np.float32)
#     vecs = []
#     start = np.asarray([0,0,0,1], dtype=np.float32).reshape(4,1)
#     for xform in xforms:
#         if xform is None:
#             continue
#         xform4x4 = np.eye(4, 4, dtype=xform.dtype)
#         xform4x4[0:3,0:4] = xform
#         refx = np.matmul(refx, xform4x4)
#         # vec = np.matmul(xform4x4, start)
#         vec = np.matmul(refx, start)
#         # input(vec.shape)
#         vecs.append(vec)
#     vecs = np.hstack(vecs)
#     input('vecs.shape: %s' % str(vecs.shape))
#     Xs =  vecs[0,:]
#     Ys =  vecs[1,:]
#     Zs =  vecs[2,:]
#
#     matplotlib.rcParams['legend.fontsize'] = 10
#
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     ax.plot(Xs, Ys, Zs, label='xforms')
#     ax.legend()
#
#     plt.show()
#
#
def plot3Dto2Dposes(xforms, camMtx, zPos=0.):
    vecs = []
    start = np.asarray([0,0,0,1], dtype=np.float32).reshape(4,1)
    for xform in xforms:
        if xform is None:
            continue
        xform4x4 = np.eye(4, 4, dtype=xform.dtype)
        xform4x4[0:3,0:4] = xform
        # xform3x3 = convert_3x4_to_3x3(xform, camMtx, zPos)
        vec4 = np.matmul(xform4x4, start)
        vec3 = np.matmul(camMtx, vec4)
        vecs.append(vec3)
    vecs = np.hstack(vecs)
    input('vecs.shape: %s' % str(vecs.shape))
    Xs =  vecs[0,:]
    Ys =  vecs[1,:]
    # Zs =  vecs[2,:]

    matplotlib.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(Xs, Ys, label='xforms')
    ax.legend()

    plt.show()


# def plot2Dposes(xforms):
#     origin = np.eye(4,4,dtype=np.float32) * 0.
#     refx = np.eye(4,4,dtype=np.float32)
#     frame2frameVecs = []
#     start = np.asarray([0,0,0,1], dtype=np.float32).reshape(4,1)
#     prevXform = origin.copy()
#     for i,xform in enumerate(xforms):
#         if xform is None:
#             continue
#
#         xform4x4 = np.eye(4, 4, dtype=xform.dtype)
#         xform4x4[0:3,0:4] = xform
#         frame2frameXform = np.matmul(xform4x4, np.linalg.pinv(prevXform))
#         refx = np.matmul(refx, frame2frameXform)
#         vec = np.matmul(refx, start)
#         frame2frameVecs.append(vec)
#         prevXform = refx.copy()
#     frame2frameVecs = np.hstack(frame2frameVecs)
#     input('frame2frameVecs.shape: %s' % str(frame2frameVecs.shape))
#     # Xs =  vecs[0,:]
#     # Ys =  vecs[1,:]
#     Xs =  frame2frameVecs[0,:]
#     Ys =  frame2frameVecs[1,:]
#     # Zs = np.zeros_like(Ys)
#     # Zs =  vecs[2,:]
#     Zs =  frame2frameVecs[2,:]
#
#     matplotlib.rcParams['legend.fontsize'] = 10
#
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     ax.plot(Xs, Ys, Zs, label='xforms')
#     ax.legend()
#
#     plt.show()
#
#
def plot2Dposes(xforms):
    refx = np.eye(4, 4, dtype=np.float32)
    vecs = []
    start = np.asarray([0, 0, 0, 1], dtype=np.float32).reshape(4, 1)
    for xform in xforms:
        if xform is None:
            continue
        xform4x4 = np.eye(4, 4, dtype=xform.dtype)
        xform4x4[0:3, 0:4] = xform
        # refx = np.matmul(refx, xform4x4)
        vec = np.matmul(xform4x4, start)
        # vec = np.matmul(refx, start)
        # input(vec.shape)
        vecs.append(vec)
    vecs = np.hstack(vecs)
    input('vecs.shape: %s' % str(vecs.shape))
    Xs = vecs[0, :]
    Ys = vecs[1, :]
    # Zs = np.zeros_like(Ys)
    Zs = vecs[2, :]

    matplotlib.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(Xs, Ys, Zs, label='xforms')
    ax.legend()

    plt.show()


def loadCamMtx34():
    camFile = 'scratch/slam/pensa_1_params.yaml'
    with open(camFile, 'r') as f:
        camParams = yaml.load(f)
    Fx = camParams['fx']
    Fy = camParams['fy']
    Cx = camParams['cx']
    Cy = camParams['cy']

    camMtx = np.asarray([
    [Fx, 0., Cx, 0.],
    [0., Fy, Cy, 0.],
    [0., 0., 1., 0.]
    ])
    return camMtx


def loadCamMtx33():
    camFile = 'scratch/slam/pensa_1_params.yaml'
    with open(camFile, 'r') as f:
        camParams = yaml.load(f)
    Fx = camParams['fx']
    Fy = camParams['fy']
    Cx = camParams['cx']
    Cy = camParams['cy']

    camMtx = np.asarray([
    [Fx, 0., Cx],
    [0., Fy, Cy],
    [0., 0., 1.]
    ])
    return camMtx

if __name__ == "__main__":
    nbFrames = 428
    path = 'localization/transformation/xforms.txt'
    xforms,_ = parseXformsFile(path, nbFrames)

    camFile = '/Users/joeliven/PensaSystems/repos/pensa-recognition/scratch/slam/pensa_1_params.yaml'
    with open(camFile, 'r') as f:
        camParams = yaml.load(f)
    Fx = camParams['fx']
    Fy = camParams['fy']
    Cx = camParams['cx']
    Cy = camParams['cy']

    camMtx = np.asarray([
    [Fx, 0., Cx, 0.],
    [0., Fy, Cy, 0.],
    [0., 0., 1., 0.]
    ])
    # camMtx = np.asarray([
    # [Fx, 0., Cx],
    # [0., Fy, Cy],
    # [0., 0., 1.]
    # ])
    # plot3Dposes(xforms)
    # plot2Dposes(xforms)
    plot3Dto2Dposes(xforms, camMtx, zPos=0.)
