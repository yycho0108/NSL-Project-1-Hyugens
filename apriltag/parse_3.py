#!/usr/bin/env python2

import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation, TimedAnimation, writers

import sys
import argparse
from utils import proc, rmat, plot_circle

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

def hide_axis(ax):
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

def lims(x, y, s=1.1, equal=False):
    xmn, xmx = np.min(x), np.max(x)
    xmd = (xmx+xmn)/2.0
    sx = (xmx - xmn)

    ymn, ymx = np.min(y), np.max(y)
    ymd = (ymx+ymn)/2.0
    sy = (ymx - ymn)

    if equal:
        sx = max(sx, sy)
        sy = max(sx, sy)

    return (xmd - sx*s/2, xmd + sx*s/2), (ymd - sy*s/2, ymd + sy*s/2)

def lim2aspect(lim):
    xs = np.abs(lim[0][1] - lim[0][0])
    ys = np.abs(lim[1][1] - lim[1][0])
    return ys/xs

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

    #np.savetxt('with_theta.csv',
    #    np.concatenate((data,th[...,np.newaxis]),axis=-1),
    #    delimiter=',',
    #    header = 't x y vx vy th'
    #    )

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

    #fig, ax = plt.subplots()
    fig = plt.figure(figsize=(8,6))

    ax = fig.add_subplot(111)
    hide_axis(ax)

    ax0 = fig.add_subplot(211)

    print len(th)
    print 'th0', np.max(th)
    print 'thmin', np.min(th)
    l0 = []
    c0 = []
    fit_t = np.exp(a) * np.exp(g*t)

    lim = lims(t, th)
    aspect = lim2aspect(lim)

    ax0.set_xlim(lim[0])
    ax0.set_ylim(lim[1])

    l0.append(Line2D(t, th, label='data', color='b'))
    l0.append(Line2D(times, peaks, marker='*', label='peaks', color='r'))
    l0.append(Line2D(t, fit_t, linestyle='--', label='fit', color='g'))
    c0.append(Ellipse((0,0), 15.0, 15.0*aspect, fc='y'))

    [ax0.add_line(l) for l in l0]
    [ax0.add_patch(c) for c in c0]

    ax0.text(0.6, 0.95, r'$ \theta(t) \approx %.2f \cdot e^{%.5f t}$' % (np.exp(a), g),
            verticalalignment='top', horizontalalignment='left',
            transform=ax0.transAxes
            )
    ax0.set_xlabel('Time (s)')
    ax0.set_ylabel(r'$\theta (deg)$')
    #ax0.set_xlim(np.min(t), np.max(t))
    #ax0.set_ylim(np.min(th), np.max(th))
    ax0.legend()#['data', 'peaks', 'fit'])
    ax0.grid()

    ## 10
    l10 = []
    c10 = []
    ax10 = fig.add_subplot(223)
    ax10.set_aspect('equal','datalim')
    ax10.set_xlabel('x (mm)')
    ax10.set_ylabel('y (mm)')

    lim = lims(np.append(x,px), np.append(y,py), equal=True)
    aspect = lim2aspect(lim)
    ax10.set_xlim(lim[0])
    ax10.set_ylim(lim[1])
    ax10.grid()

    l10.append(Line2D(x,y))
    c10.append(Ellipse((0,0), 10.0, 10.0*aspect, fc='r'))
    px = np.mean(x)
    py = np.min(y) + 343
    pivot = (px, py)
    l10.append(plot_circle(pivot, 5, ax=ax10))

    [ax10.add_line(l) for l in l10]
    [ax10.add_patch(c) for c in c10]


    # ax11
    ax11 = fig.add_subplot(224)

    lim = lims(x, y, equal=False)
    aspect = lim2aspect(lim)
    ax11.set_xlim(lim[0])
    ax11.set_ylim(lim[1])
    ax11.grid()

    l11 = []
    c11 = []
    l11.append(Line2D(x,y))
    c11.append(Ellipse((0,0), 2.0, 2.0*aspect, fc='r'))
    [ax11.add_line(l) for l in l11]
    [ax11.add_patch(c) for c in c11]

    ax11.set_xlabel('x (mm)')
    ax11.set_ylabel('y (mm)')

    print 'gamma : {}'.format(g)

    #plt.plot(peaks)
    #plt.rc('text', usetex=True)
    ax.set_title('Damping Behavior Without Escapement')
    print 'g', g

    if opts.animate:
        def animate(i):
            ti = t[i]
            txi = np.argmin(np.abs(times - ti))
            l0[0].set_data(t[:i], th[:i])
            l0[1].set_data(times[:txi], peaks[:txi])
            l0[2].set_data(t[:i], fit_t[:i])
            c0[0].center = (t[i], th[i])

            l10[0].set_data(x[:i], y[:i])
            c10[0].center = (x[i], y[i])

            l11[0].set_data(x[:i], y[:i])
            c11[0].center = (x[i], y[i])
            return l0+c0+l10+c10+l11+c11

        ani = FuncAnimation(fig, animate, frames=len(x), blit=True, interval=10.)
        
        Writer = writers['ffmpeg']
        writer = Writer(fps=120, metadata=dict(artist='Me'), bitrate=1800)
        ani.save('parse_3.mp4', writer=writer)

        #ani = TimedAnimation(fig, blit=True)
        #ani._draw_frame = 

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
    parser.add_argument('--animate', type=str2bool, default='False')

    opts = parser.parse_args(sys.argv[1:])
    if opts.proc:
        # processing
        with open(opts.filename) as f:
            process(f, opts)
    else:
        # plotting
        plot(opts.filename)
