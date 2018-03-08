#!/usr/bin/env python2

import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, TimedAnimation, writers

from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints as MSS

import sys
import argparse

def v2vx(v):
    v1,v2,v3 = v
    return np.reshape([0, -v3, v2, v3, 0, -v1, -v2, v1, 0], (3,3)).astype(np.float32)

def rmat(v0, v1):
    # returns 3d rotation Matrix M, such that
    # M * uvec(v0) = uvec(v1)

    u = v0
    v = v1

    axis = np.cross(u, v)
    axis /= np.linalg.norm(axis)
    ax,ay,az = axis
    th = np.arccos(np.dot(u, v))
    c = np.cos(th)
    _c = 1.0 - c
    s = np.sin(th)

    M = (1.0 - c) * np.outer(axis, axis)
    M += np.reshape([c, -s*az, s*ay, s*az, c, -s*ax, -s*ay, s*ax, c],(3,3))
    return np.asarray(M, dtype=np.float32)

def project2d(vs, ns):
    # vs = vectors
    # ns = normal vectors
    Rs = [rmat(n, np.asarray([0,0,1])) for n in ns]
    return np.asarray([R.dot(v) for (R,v) in zip(Rs,vs)], dtype=np.float32)

def reorder(e1, id0):
    id1 = np.int32([k[0] for k in e1]) # say 31, _, 25, need to be organized as [?, 0, 2]

    midx = []
    idx = []
    res = []
    for i0, _id0 in enumerate(id0):
        for i1, _id1 in enumerate(id1):
            if _id0==_id1:
                res.append(e1[i1][1:]) # skip id
                break
        else:
            # id not found
            res.append(None)
            midx.append(i0)
    return np.int32(midx), res

def hx(x):
    #x,y,vx,vy
    return x[:3]

def fx(x, dt):
    x,y,z,vx,vy,vz = x
    x=x+vx*dt
    y=y+vy*dt
    z+y+vz*dt
    return (x,y,z,vx,vy,vz)

def main(f, opts):

    fps = 60
    dt = 1.0/fps

    k = f.readlines()[1000:2000]
    k = [np.asarray([r.split(',') for r in e.split('|')[:-1]], dtype=np.float32) for e in k]
    #k = [id,x,y,z,id,x,y,z,id,x,y,z]

    # process & match ids, etc...

    fpt = MSS(6,1e-3,2,0.0)

    ukfs = [UKF(6,3,dt,hx=hx,fx=fx,points=fpt) for _ in range(4)]

    for i, u in enumerate(ukfs):
        u.x = np.zeros(6, dtype=np.float32)
        u.x[:3] = k[0][i][1:]
        #u.P = (5.0)**2 * np.eye(6) # initial cov
        u.P = np.square(np.diag([5.0, 5.0, 5.0, 10.0, 10.0, 10.0]))
        u.R = (5.0)**2 * np.eye(3) # measurement cov, ~ +-5mm std.
        u.Q = (3.0)**2 * np.eye(6) # process cov ~ +-3mm std.

    ids = np.int32([l[0] for l in k[0]]) # say 15, 25, 31

    k2 = []
    cnt=0

    vs = []
    for c, e in enumerate(k[1:]):
        if c % 100 == 0:
            print c
        [u.predict(dt) for u in ukfs]
        midx, obs = reorder(e, ids)
        if midx:
            # only update applicable ones
            for i in range(4):
                if i in midx:
                    continue
                ukfs[i].update(obs[i])
        else:
            # correct based on estimates
            [u.update(e) for (u,e) in zip(ukfs, obs)]

        entry = []
        ventry = []
        for u in ukfs:      
            entry.append(u.x[:3])
            ventry.append(u.x[3:])
        k2.append(np.float32(entry))
        vs.append(np.float32(ventry))

    vs = np.float32(vs)
    print np.shape(vs)
    print np.max(vs, axis=0)
    print np.min(vs, axis=0)

    k2 = np.asarray(k2, dtype=np.float32) # k2 = [n, i, p]
    print np.shape(k2)

    np.negative(k2[:,:,1], k2[:,:,1]) # flip y

    plt.plot(k2[:,0,0], k2[:,0,1]) #plot x-y
    plt.show()
    return

    t = (1./fps) * np.arange(len(k2)) # 60fps


    # 3-4-5 : defines a plane
    x3, y3, z3 = k2[:,2].T
    x4, y4, z4 = k2[:,3].T
    x5, y5, z5 = k2[:,4].T

    v1 = np.subtract(k2[:,3], k2[:,2])
    v2 = np.subtract(k2[:,3], k2[:,4])
    n = np.cross(v1,v2)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)

    # align to normal plane
    for i in range(6):
        k2[:,i] = project2d(k2[:,i], n)
    k2 -= k2[:,np.newaxis, 5] # subtract track reference, for camera motion

    # align to horizontal
    v1 = np.subtract(k2[:,3], k2[:,2])
    v1[:,2] = 0 # set z=0 to force rotation in z axis
    v1 /= np.linalg.norm(v1, axis=-1, keepdims=True)
    Rs = [rmat(_n, np.asarray([1,0,0])) for _n in v1]
    for i in range(6):
        k2[:,i] = np.asarray([R.dot(v) for (R,v) in zip(Rs,k2[:,i])], dtype=np.float32)
    k2 -= k2[:,np.newaxis, 5] # subtract track reference, for camera motion

    k2 = k2[:,:,:2] # convert to 2d

    plt.rc('font', family='serif')

    ani = SubplotAnimation(k2,t)

    if opts.outfile:
        Writer = writers['ffmpeg']
        writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)
        ani.save('meh3.mp4', writer=writer)

    plt.show()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Process Tag Data, and produce coordinate data and plots'
            )
    parser.add_argument('filename', type=str)#, nargs='1')
    parser.add_argument('--outfile', type=str, nargs='?', const=True, default='', help='Video output file')

    opts = parser.parse_args(sys.argv[1:])
    with open(opts.filename) as f:
        main(f, opts)
