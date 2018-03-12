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

from utils import proc, rmat

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

def process(f, opts):
    fps = 60
    dt = 1.0/fps

    # Data Format : [id,x,y,z | id,x,y,z | id,x,y,z ...]
    raw_data = f.readlines()[1:]
    raw_data = [np.asarray([r.split(',') for r in e.split('|')[:-1]], dtype=np.float32) for e in raw_data]
    #raw_data = raw_data[2000:3000]

    # Process & match ids, etc...
    ids = np.int32([l[0] for l in raw_data[0]]) # say 15, 25, 31
    pos,vel = proc(raw_data, ids, dt) 

    # flip y ...
    np.negative(pos[:,:,1], pos[:,:,1])
    np.negative(vel[:,:,1], vel[:,:,1])

    # remove drift over time ...
    ctr = np.mean(pos[:,1:,:], axis=1, keepdims=True) # center of three triangles
    print np.shape(ctr)
    pos -= ctr

    # calculate normal vector
    v1 = np.subtract(pos[:,1], pos[:,2])
    v2 = np.subtract(pos[:,1], pos[:,3])
    n = np.cross(v1,v2)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)

    # align all to normal plane
    for i in range(4):
        pos[:,i] = project2d(pos[:,i], n)
        vel[:,i] = project2d(vel[:,i], n)

    pos = pos[...,:2] # remove z component
    vel = vel[...,:2]

    # get pivot point

    #sample_idx = np.random.choice(len(pos), size=1000, replace=False)
    #pivot = get_pivot_2(pos[sample_idx,0])
    px = np.mean(pos[:,0])
    py = np.min(pos[:,1]) + 343
    pivot = (px, py)

    x, y = pos[:,0].T

    # get angle
    dx = x - pivot[0]
    dy = pivot[1] - y
    th = np.arctan2(dx, dy)

    # time
    t = dt * np.arange(len(pos)) # 60fps
    t = t[..., np.newaxis] # add axis for concatenating
    
    data = np.concatenate((t, pos[:,0], vel[:,0]), axis=-1)

    if opts.outfile:
        #np.save(opts.outfile, (t, th))
        np.savetxt(opts.outfile,
            data,
            delimiter=',',
            header = 't x y vx vy'
            )

    #plt.plot(t[offset:], th[offset:])
    #plt.show()

    #plt.plot(pos[:,0,0], pos[:,0,1])
    #plt.show()

def plot(f):
    fps = 60
    dt = 1.0 / fps
    offset = 10 * fps # wait ~10 sec. for stabilization
    period = int(1.5 * fps)

    data = np.loadtxt(f, delimiter=',')
    t,x,y,vx,vy = data.T

    px = np.mean(x)
    py = np.min(y)+343.
    dx = x - px
    dy = py - y

    th = np.arctan2(dx, dy)
    th = np.rad2deg(th)

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
            print '!!'
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
    print len(th)
    print 'th0', np.max(th)
    print 'thmin', np.min(th)
    plt.plot(t, th)
    plt.plot(times, peaks, '*')
    plt.plot(t, np.exp(a)*np.exp(g*t), '--')

    print 'gamma : {}'.format(g)

    #plt.plot(peaks)
    #plt.rc('text', usetex=True)
    plt.title('Damping Behavior Without Escapement')
    plt.xlabel('Time (s)')
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
            process(f, opts)
    else:
        # plotting
        plot(opts.filename)
