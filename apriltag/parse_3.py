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
from scipy.signal import find_peaks_cwt

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

def get_pivot_2(pts, r=343.):
    # r = 343
    n,m = np.shape(pts)
    if n==2:
        pts = np.transpose(pts)
    n = len(pts)
    res = []
    for i in range(n):
        for j in range(i+1,n):
            p1, p2 = pts[i], pts[j]
            dp = np.subtract(p2,p1)
            d = np.linalg.norm(dp)
            a = d/2.
            h = np.sqrt(np.abs(r**2-a**2))
            if dp[0] >= 0:
                x3 = p2[0] - h * dp[1] / d
                y3 = p2[1] + h * dp[0] / d
            else:
                x3 = p2[0] + h * dp[1] / d
                y3 = p2[1] - h * dp[0] / d
            res.append((x3,y3))
    return np.mean(res, axis=0)

def proc(f, opts):
    fps = 60
    dt = 1.0/fps

    # Data Format : [id,x,y,z | id,x,y,z | id,x,y,z ...]
    raw_data = f.readlines()[1:]
    raw_data = [np.asarray([r.split(',') for r in e.split('|')[:-1]], dtype=np.float32) for e in raw_data]


    # Process & match ids, etc...
    fpt = MSS(6, 1e-3, 2, 0.0)
    ukfs = [UKF(6,3,dt,hx=hx,fx=fx,points=fpt) for _ in range(4)]
    for i, u in enumerate(ukfs):
        u.x = np.zeros(6, dtype=np.float32)
        u.x[:3] = raw_data[0][i][1:]
        u.P = np.square(np.diag([5.0, 5.0, 5.0, 10.0, 10.0, 10.0])) # initial cov
        u.R = 0.005 * np.eye(3) #(5.0)**2 * np.eye(3) # measurement cov, ~ +-5mm std.
        u.Q = (1.0)**2 * np.eye(6) # process cov ~ +-3mm std.

    ids = np.int32([l[0] for l in raw_data[0]]) # say 15, 25, 31

    l = len(raw_data) - 1 # skip the first one, so...
    pos = []
    vel = []
    for c, e in enumerate(raw_data[1:]):
        if c % 100 == 0:
            print '{}/{}\r'.format(c, l)
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
        pos.append(np.float32(entry))
        vel.append(np.float32(ventry))

    vel = np.float32(vel)
    pos = np.float32(pos)

    #print np.shape(vel)
    #print np.max(vel, axis=0)
    #print np.min(vel, axis=0)

    pos = np.asarray(pos, dtype=np.float32) # k2 = [n, i, p]

    # flip y ...
    np.negative(pos[:,:,1], pos[:,:,1])
    np.negative(vel[:,:,1], vel[:,:,1])

    # remove drift over time ...
    ctr = np.mean(pos[:,1:,:], axis=1, keepdims=True) # center of three triangles
    print np.shape(ctr)
    pos -= ctr

    # simple 2d viz for testing
    #plt.plot(pos[:,0,0], pos[:,0,1]) #plot x-y
    #m = pos
    #plt.plot(m[:,1,0], m[:,1,1]) #plot x-y
    #v1 = np.var(m[:,1], axis=0)
    #v2 = np.var(m[:,2], axis=0)
    #v3 = np.var(m[:,3], axis=0)
    #print np.mean([v1,v2,v3], axis=0)
    #plt.show()

    t = (1./fps) * np.arange(len(pos)) # 60fps

    # calculate normal vector
    v1 = np.subtract(pos[:,1], pos[:,2])
    v2 = np.subtract(pos[:,1], pos[:,3])
    n = np.cross(v1,v2)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)

    # align to normal plane
    for i in range(4):
        pos[:,i] = project2d(pos[:,i], n)
    pos = pos[:,:,:2] # remove z component

    # get pivot point
    sample_idx = np.random.choice(len(pos), size=1000, replace=False)
    pivot = get_pivot_2(pos[sample_idx,0])

    x, y = pos[:,0].T

    # get angle
    dx = x - pivot[0]
    dy = pivot[1] - y
    th = np.arctan2(dx, dy)

    if opts.outfile:
        np.save(opts.outfile, (t, th))
    #plt.plot(t[offset:], th[offset:])
    #plt.show()

    #plt.plot(pos[:,0,0], pos[:,0,1])
    #plt.show()

def plot(f):
    fps = 60
    dt = 1.0 / fps
    offset = 10 * fps # wait ~10 sec. for stabilization
    period = int(1.1 * fps)

    fin = offset + 5 * fps
    t, th = np.load(f)

    th = np.rad2deg(th)

    w = np.diff(th) / dt # or thereabouts, anyway.
    print np.mean(np.abs(w))
    #peaks = find_peaks_cwt(th, range(window-5,window+5))

    # first peak. also defines amplitude
    p = np.argmax(th)
    pidx = [p]
    while True:
        p0 = p + period - period/2
        p1 = p + period + period/2
        try:
            p = p0 + np.argmax(th[p0:p1])
            pidx.append(p)
        except:
            break

    #pidx = np.divide(peaks[:-1], p) # peaks = p0*e^(-g*t)

    peaks = th[pidx]
    times = t[pidx]
    sel = (peaks > 0)
    peaks = peaks[sel]
    times = times[sel]

    lpeaks = np.log(peaks) # pk(t) = -g*t + log(p0)
    g, a = np.polyfit(times, lpeaks, 1)

    fig, ax = plt.subplots()
    plt.plot(t, th)
    plt.plot(times, peaks, '*')
    plt.plot(t, np.exp(a)*np.exp(g*t), '--')

    print 'gamma : {}'.format(g)

    #plt.plot(peaks)
    #plt.rc('text', usetex=True)
    plt.title('Damping Behavior Without Escapement')
    plt.xlabel('Time (ms)')
    plt.ylabel(r'$\theta (deg)$')
    plt.legend(['data', 'peaks', 'fit'])
    print 'g', g
    plt.text(0.3, 0.95, r'$ \theta(t) \approx %.2f \cdot e^{%.5f t}$' % (np.exp(a), g),
            verticalalignment='top', horizontalalignment='left',
            transform=ax.transAxes
            )
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
    parser.add_argument('--proc', type=str2bool, default='False')
    parser.add_argument('--outfile', type=str, nargs='?', const=True, default='', help='Data output file')

    opts = parser.parse_args(sys.argv[1:])
    if opts.proc:
        # processing
        with open(opts.filename) as f:
            proc(f, opts)
    else:
        # plotting
        plot(opts.filename)
