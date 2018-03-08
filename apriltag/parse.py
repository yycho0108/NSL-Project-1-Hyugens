#!/usr/bin/env python2

import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, TimedAnimation, writers

import sys
import argparse

class Tag(object):
    def __init__(self, data):
        self._id = data[0]
        self._x = data[1:]
    def match(self, o):
        if self._id != o._id:
            return np.inf
        else:
            return np.linalg.norm(self._x - o._x)

def reorder(e0, e1):
    """
    Use reorder() iteratively in case ids have duplicates.
    Figures out tag assignment based on previous positions.
    """
    t0 = [Tag(i) for i in e0]
    t1 = [Tag(i) for i in e1]

    n0 = len(t0)
    n1 = len(t1)

    cost = np.zeros((n0,n1), dtype=np.float32)
    for i in range(n0):
        for j in range(n1):
            cost[i,j] = t0[i].match(t1[j])

    #cost = np.square(np.subtract.outer(e0, e1))
    #print cost.shape
    i,j = linear_sum_assignment(cost)

    #if (n0 != 3 or n1 != 3):
    #    print cost
    #    print n0, n1
    #    print i,j

    # t0[i] => t1[j]
    return e1[j]

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


class SubplotAnimation(TimedAnimation):
    def __init__(self, k2, t):
        fig = plt.figure()
        fig.set_dpi(100)
        fig.set_size_inches(12, 6)

        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 4, sharex=ax2)

        self.k2 = k2
        self.t = t

        x1, y1 = k2[:,0].T
        x2, y2 = k2[:,1].T
        x3, y3 = k2[:,2].T
        x4, y4 = k2[:,3].T

        self.x1, self.y1 = x1,y1
        self.x2, self.y2 = x2,y2

        self.th1 = np.arctan2(x1-x3,(y3-25.4)-y1)
        self.th2 = np.arctan2(x2-x4,(y4-25.4)-y2)

        # ax1 
        ax1.set_xlabel('x(mm)')
        ax1.set_ylabel('y(mm)')
        ax1.set_aspect('equal', 'datalim')
        self.ps1 = [Line2D([],[]) for _ in range(6)]

        self.pc1 = plt.Circle((0,0), 10, fc='r')
        self.pc2 = plt.Circle((0,0), 10, fc='b')

        [ax1.add_line(l) for l in self.ps1]
        ax1.add_patch(self.pc1)
        ax1.add_patch(self.pc2)

        ax1.set_xlim(np.min(k2[:,:,0]), np.max(k2[:,:,0]))
        ax1.set_ylim(np.min(k2[:,:,1]), np.max(k2[:,:,1]))
        ax1.set_aspect('equal','datalim')

        # ax2
        #ax2.set_xlabel('t(sec)')
        ax2.set_ylabel(r'$\theta_l$(rad)')
        self.p2 = Line2D([],[])
        ax2.add_line(self.p2)
        ax2.set_xlim(np.min(t), np.max(t))
        ax2.set_ylim(np.min(self.th1), np.max(self.th1))
        plt.setp(ax2.get_xticklabels(), visible=False)
        self.ax2 = ax2

        # ax3
        ax3.set_xlabel('t(sec)')
        ax3.set_ylabel(r'$\theta_r$(rad)')
        self.p3 = Line2D([],[])
        ax3.add_line(self.p3)
        ax3.set_xlim(np.min(t), np.max(t))
        ax3.set_ylim(np.min(self.th2), np.max(self.th2))
        self.ax3 = ax3

        TimedAnimation.__init__(self, fig, interval=1000/60., blit=True, repeat=False)

    def _draw_frame(self, n):
        for i in range(6):
            (x,y) = self.k2[:n,i].T
            self.ps1[i].set_data(x,y)

        self.pc1.center = self.x1[n], self.y1[n]
        self.pc2.center = self.x2[n], self.y2[n]

        i0 = max(0, int(n - 0.5*60))
        i1 = min(len(self.t), int(n + 0.5*60))

        self.p2.set_data(self.t[i0:i1], self.th1[i0:i1])
        self.ax2.set_xlim(min(self.t[i0:i1]), max(self.t[i0:i1]))
        self.ax2.set_ylim(min(self.th1[i0:i1]), max(self.th1[i0:i1]))

        self.p3.set_data(self.t[i0:i1], self.th2[i0:i1])
        self.ax3.set_xlim(min(self.t[i0:i1]), max(self.t[i0:i1]))
        self.ax3.set_ylim(min(self.th2[i0:i1]), max(self.th2[i0:i1]))
        
        self._drawn_artists = self.ps1 + [self.p2, self.p3] + [self.pc1, self.pc2]

    def new_frame_seq(self):
        return iter(range(self.t.size))

    def _init_draw(self):
        lines = self.ps1 + [self.p2, self.p3]
        for l in lines:
            l.set_data([], [])

def main(f, opts):

    fps = 60
    k = f.readlines()[1:]
    k = [np.asarray([r.split(',') for r in e.split('|')[:-1]], dtype=np.float32) for e in k]
    #k = [id,x,y,z,id,x,y,z,id,x,y,z]

    # process & match ids, etc...
    e0 = k[0]
    k2 = []
    for e in k:
        e0n = reorder(e0, e)
        if len(e0n) < 3:
            # TODO: fill - with - anticipated
            continue
        e0 = e0n
        k2.append(np.copy(e0[:,1:]))
    #print len(k)
    k2 = np.asarray(k2)
    np.negative(k2[:,:,1], k2[:,:,1]) # flip y

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
